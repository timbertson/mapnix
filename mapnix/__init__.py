from __future__ import print_function
import sys
import os
import optparse
import subprocess
import re
import errno
import logging
import tempfile, shutil
import contextlib
import pipes
from xml.etree import ElementTree
import json
from distutils import spawn
import collections
import threading
# from pdb import set_trace as pdb_trace
from .nixops import nix_expr

log = logging.getLogger('mapnix')

here = os.path.dirname(__file__)
nixlib = os.path.join(here, "nix-defs")

def destroy(node):
	if not node.destroy():
		raise AssertionError("Failed to destroy %r" % (node,))

def err(msg):
	print(msg, file=sys.stderr)

def heading(msg):
	print("\n** " + msg)

def cancel():
	err("Cancelled.")
	sys.exit(1)

ROOT_DISK_ID = 'root'
DEFAULT_HOOK = {'order':{'parallel':0, 'batch':None, 'sort':None}}
SHOW_TRACE = False
DUMP_SIZE_LIMIT = 1024 * 4

# in order to ensure that resources are created / deleted in
# the right order relative to their parent/children, we sort
# diffs by their operation
class Object(object): pass
Order = Object()
Order.Noop = 0

# NOTE: image creation can take VERY LONG and may fail,
# so we want to do it before any drastic changes
Order.CreateImageStorage = Order.Noop+1
Order.CreateImage        = Order.CreateImageStorage+1

Order.DetachDisk         = Order.CreateImage+1
Order.DeleteFirewall     = Order.CreateImage+1
Order.DeleteMachine      = Order.DetachDisk+1
Order.DeleteDisk         = max(Order.DetachDisk, Order.DeleteMachine)+1
Order.DeleteAddress      = Order.DeleteMachine+1
Order.DeleteNetwork      = max(Order.DeleteMachine, Order.DeleteFirewall)+1
Order.DeleteImage        = Order.DeleteMachine+1
Order.DeleteImageStorage = Order.DeleteImage+1


Order.Modify             = 100 # self-contained modification (no dependencies)
assert Order.Modify > Order.DeleteImage

Order.CreateNetwork      = Order.Modify+1
Order.CreateFirewall     = Order.CreateNetwork+1
Order.CreateDisk         = Order.Modify+1
Order.CreateAddress      = Order.CreateNetwork+1
Order.CreateMachine      = max(Order.CreateAddress, Order.CreateDisk, Order.CreateNetwork)+1

Order.AttachDisk         = Order.CreateMachine+1
Order.AttachAddress      = max(Order.AttachDisk,Order.CreateNetwork)+1

Order.AwaitMachine       = Order.AttachAddress+1


def prompt(msg, valid_responses):
	while True:
		response = raw_input(msg)
		if response in valid_responses:
			return response
		else:
			err("Invalid option - try again")

def groupby(items, key):
		groups = []
		# like itertools.groupby, but doesn't care about ordering,
		# and doesn't require hashable keys
		for item in items:
			group_key = key(item)
			for group in groups:
				if group[0] == group_key:
					group[1].append(item)
					break
			else:
				groups.append((group_key, [item]))
		return groups

def nix_cmd(cmd):
	if SHOW_TRACE:
		cmd = cmd + ['--show-trace']
	return cmd

def nix_eval_cmd(args, xml=True):
	cmd = nix_cmd([
		'nix-instantiate',
		'--eval',
		'--strict',
	]+args)
	if xml:
		cmd.append('--xml')
	return cmd

def nix_build_cmd(dest, args):
	return nix_cmd([
		'nix-build',
		'-o', dest,
	]+args)


def ignore(*_): pass # for pychecker
def TODO(): raise AssertionError("TODO")
def NOT_IMPLEMENTED(*_): raise AssertionError("not implemented")

class BootstrapImage(object):
	name_prefix = "mapnix-"
	def __init__(self, identity, builder):
		self.identity = identity
		self.builder = builder
		self.name = 'mapnix-%s' % (identity)
		self.filename = self.name + '.tar.gz'
		self._built = False
	
	def build(self):
		if self._built is not None:
			self._built = self.builder()
		return self._built

def rm_rf(path):
	try:
		os.remove(path)
	except OSError as e:
		if e.errno == errno.ENOENT: return
		shutil.rmtree(path)

def consistent(set, desc):
	it = iter(set)
	rv = next(it)
	while True:
		try:
			item = next(it)
		except StopIteration: break
		if item != rv:
			raise AssertionError("inconsistent values for `%s`" % (desc,))
	return rv

SENTINEL = object()
def getpath(obj, path, default=SENTINEL):
	v = obj
	if isinstance(path, basestring): path = path.split('.')
	for key in path:
		try:
			v = v[key]
		except KeyError:
			if default is SENTINEL: raise
			return default
		except TypeError as e:
			raise TypeError("Error getting %r from %r: %s" % (key, v, e))
	if v is None:
		if default is SENTINEL: raise TypeError("value at %r is None" % (key,))
		else: return default
	return v

@contextlib.contextmanager
def atomic_write(path):
	tmp = path + '.tmp'
	rm_rf(tmp)
	try:
		yield tmp
	except Exception as e:
		rm_rf(tmp)
		raise e
	rm_rf(path)
	os.rename(tmp, path)

def assert_unique(iter, desc):
	seen = set()
	for item in iter:
		if item in seen:
			raise AssertionError("duplicate %s: %s" % (desc, item))
		seen.add(item)
	return iter

key_safe_re = re.compile('^[-_a-zA-Z0-9]+$')
def key_safe(key):
	if re.match(key_safe_re, key):
		return key
	else:
		raise AssertionError("key contains invalid characters: %s" % (key,))

def dump_cmd(cmd):
	log.debug("+ " + " ".join(cmd))

def run_cmd(cmd, *a, **k):
	dump_cmd(cmd)
	return subprocess.check_call(cmd, *a, **k)

def run_output(cmd, *a, **k):
	dump_cmd(cmd)
	return subprocess.check_output(cmd, *a, **k)

def xml2py(string, basedir, file_digests=False):
	# when file_digests is true, return the nix hash of each referenced
	# file expression, rather than its path. This is only useful for
	# computing identity, as the original file contents will not be accessible
	def convert(elem):
		def only_child(elem):
			assert len(elem) == 1, "expected 1 child of <%s>, got %s" % (elem.tag, len(elem))
			return next(iter(elem))

		def no_attrs(elem, attrib=None):
			if attrib is None: attrib = elem.attrib
			assert not attrib, "<%s> has unexpected attributes: %r" % (elem.tag, attrib)
		
		def primitive(elem):
			attrib = elem.attrib.copy()
			rv = attrib.pop('value')
			no_attrs(elem, attrib)
			return rv

		if elem.tag == 'expr':
			no_attrs(elem)
			return convert(only_child(elem))

		elif elem.tag == 'attrs':
			no_attrs(elem)
			rv = {}
			for child in elem:
				assert child.tag == 'attr'
				rv[child.attrib['name']] = convert(only_child(child))
			return rv

		elif elem.tag == 'list':
			no_attrs(elem)
			return list(map(convert, elem))

		elif elem.tag == 'string':
			return primitive(elem)

		elif elem.tag == 'path':
			if file_digests:
				digest = run_output([
					'nix-hash', os.path.join(basedir, primitive(elem))
				]).strip()
				assert len(digest) > 0, "empty digest"
				return digest
			else:
				return os.path.join(basedir, primitive(elem))

		elif elem.tag == 'int':
			return int(primitive(elem))

		elif elem.tag == 'bool':
			b = primitive(elem)
			if b == 'true': return True
			elif b == 'false': return False
			else: raise AssertionError("Unknown boolean value: " + b)
		
		elif elem.tag == 'null':
			no_attrs(elem)
			return None

		else: assert False, "Unknown xml tag: " + elem.tag

	return convert(ElementTree.fromstring(string))

PROVIDERS = {}

def cached_property(fn):
	cached = []
	def _(self):
		if not cached:
			cached.append(fn(self))
		return cached[0]
	return property(_)

class Machine(object):
	has_really_fast_connection = False
	def __init__(self, id, state, config):
		self.id = id
		self.state = state
		self.config = config
		self.user = getpath(self.config, ['ssh','user'], 'root')
		self.target = (
			self.user + '@' +
			getpath(self.config, ['ssh', 'host'], self.state['publicAddress'])
		)
		self.ssh = ['ssh', self.target]
	
	def __repr__(self): return '<Machine %s>' % (self.id,)
	
	def run_ssh(self, cmd, check=True, output=False):
		dump_cmd(self.ssh + ['--',cmd])
		if check:
			runner = subprocess.check_output if output else subprocess.check_call
			return runner(self.ssh + ['--', cmd])
		else:
			assert not output, "not supported"
			return subprocess.Popen(self.ssh + ['--', cmd]).wait()
	
	def copy_closure(self, path, cwd):
		"""Copy a closure to this machine."""
		# TODO: Implement copying between cloud machines, as in the Perl
		# version.

		# It's usually faster to let the target machine download
		# substitutes from nixos.org, so try that first.
		if not self.has_really_fast_connection:
			closure = run_output(["nix-store", "-qR", path]).splitlines()
			self.run_ssh("nix-store -j 4 -r --ignore-unknown " + ' '.join(closure), check=False)

		# Any remaining paths are copied from the local machine.
		cmd = ["nix-copy-closure", "--to", self.target, path]
		env = dict(os.environ)
		# env['NIX_SSHOPTS'] = ' '.join(ssh._get_flags() + ssh.get_master().opts)
		sysconfDir = self.config.get('sysconfDir', None)
		if sysconfDir is not None:
			env['NIX_CONF_DIR'] = os.path.join(cwd, sysconfDir)
			log.debug("Setting NIX_CONF_DIR=" + env['NIX_CONF_DIR'])
		if not self.has_really_fast_connection:
			cmd.insert(1, '--gzip')

		if self.config.get('signStore'):
			cmd.insert(1, '--sign')

		run_cmd(cmd, env=env, cwd=cwd)
	
	def gather_facts(self):
		spec = self.config['gatherFacts']
		if spec is None:
			return None
		with open(spec['module']) as f:
			expr = f.read()

		# wrap `expr` in an explicit invocation, otherwise our
		# facts resolve to a <lambda>
		expr = 'let fn = ('+expr+'); in {_argv}: { result = builtins.toJSON (fn _argv); }';
		args = nix_expr.py2nix(spec.get('args', {}))

		cmd = nix_eval_cmd([
			'--expr', expr,
			'--arg', '_argv', args,
			'-A', 'result',
		], xml=False)

		# dump_cmd(cmd)
		output = self.run_ssh(' '.join(map(pipes.quote, cmd)), output=True)
		# XXX we could just return a plain nix expression, if we
		# could splice that directly into deployment.state
		# err(output)

		# it's actually a nix-expr of a JSON string, but the string syntax overlaps
		facts = json.loads(output)
		facts = json.loads(facts)
		# facts = xml2py(output, '/')

		# XXX document & maybe split up from facts, which are otherwise entirely user-driven
		# get the IP addreses (XXX this depends on `ip` being installed...)
		if spec.get('networkAddress'):
			ip_output = self.run_ssh('ip -o addr', output=True).strip()
			ips = collections.defaultdict(lambda: {})
			typekeys = { 'inet': 'ipv4', 'inet6':'ipv6'}
			for line in ip_output.splitlines():
				line = line.split()
				iface = line[1]
				if iface.startswith('veth'):
					continue
				type = line[2]
				try:
					type = typekeys[type]
				except KeyError:
					log.warn("Skipping unknown inet type %s", type)
					continue

				addr = line[3].split('/',1)[0]
				if type in ips[iface]:
					log.warn("Multiple %s addresses found for %s; ignoring" %(type, iface))
				else:
					ips[iface][type] = addr
			facts['networkAddress'] = ips
		return facts

	def activate_config(self, path, force):
		prefix=''
		if self.user != 'root':
			prefix='sudo '

		profile_path = "/nix/var/nix/profiles/system"
		# ALWAYS activate boot config, and then try to switch.
		# If the switch fails with exit status 100, we know the
		# new config will be active after a reboot
		activate = """
			{prefix} NIX_REMOTE=daemon nix-env -p {profile} --set "{path}"
			{prefix} {profile}/bin/switch-to-configuration boot
			{prefix} {profile}/bin/switch-to-configuration switch
		"""

		path = os.path.realpath(path)

		if not force:
			activate = """
				if [ "{profile}" -ef "{path}" ]; then
					echo "NOTE: not re-activating identical config" >&2
				else
					""" + activate + """
				fi
			"""

		ssh_cmd = "\n".join(["set -eu", activate]).format(
			prefix=prefix,
			profile=profile_path,
			path=path,
		)
		status = self.run_ssh(ssh_cmd, check=False)
		if status != 0:
			if status == 100:
				raise RuntimeError("TODO: reboot machine (for now, you must do it manually)")
			else:
				raise RuntimeError("Activation script failed with status %s" %(status,))

		log.info("activation finished successfully")

class Deployment(object):
	def __init__(self, path):
		self.path = os.path.abspath(path)
		assert os.path.exists(self.path)
		self.cwd = os.path.dirname(self.path)
		self.state = {}
		self.storage_dir = os.path.join(self.cwd, '.mapnix', os.path.basename(self.path)) 

	@property
	def state_path(self):
		return os.path.join(self.storage_dir, 'state.nix')

	def make_storage_dir(self):
		try:
			os.makedirs(self.storage_dir)
		except OSError as e:
			if e.errno != errno.EEXIST: raise
		return self.storage_dir
	
	def add_state(self, path, val, default=False):
		log.debug("setting state %s=%r", ".".join(path), val)
		state = self.state
		for key in path[:-1]:
			if key not in state:
				state[key] = {}
			state = state[key]
		key = path[-1]
		if key in state:
			if default: return
			else:
				log.warn("Overriding existing state at %s",
					".".join(path))
		state[key] = val

	def save_state(self):
		log.debug("Saving nix state: %r", self.state)
		serialized = nix_expr.py2nix(self.state)
		# log.debug("Saving nix state: %r", serialized)
		with open(self.state_path, 'w') as f:
			f.write(serialized)

	def load_cached_state(self):
		output = run_output(nix_eval_cmd([
			self.state_path,
		], xml=True))
		self.state = xml2py(output, self.cwd)
	
	# see /home/tim/dev/scratch/nixops/nixops/deployment.py
	@cached_property
	def nix_expr(self):
		# flags = list(itertools.chain(*[["-I", x] for x in (self.extra_nix_path + self.nix_path)])) + self.extra_nix_flags
		# flags.extend(["-I", "nixops=" + self.expr_path])
		# return flags
		output = run_output(nix_eval_cmd(self._eval_args(state=False, attrib='info'), xml=True))
		# err(output)
		# json_str = json.loads(output)
		# err(repr(json_str))
		# assert isinstance(json_str, unicode)
		# rv = json.loads(json_str)
		# err(repr(rv))
		rv = xml2py(output, self.cwd)
		# check various uniqueness requirements here:
		provider = getpath(rv, ['provider', 'type'])

		def check_disks():
			def all_disk_ids():
				for name, machine in rv['machines'].items():
					disk_ids = [
						disk['id'] for disk
						in getpath(machine, [provider, 'disks'], [])
					]

					assert_unique(disk_ids, 'disk IDs for %s' % (name,))
					for id in disk_ids:
						if id == ROOT_DISK_ID: continue
						yield id
			assert_unique(all_disk_ids(), 'disk ID')

		def check_networks():
			seen = {}
			for name, machine in rv['machines'].items():
				net = getpath(machine, [provider, 'network'], None)
				if net is not None:
					id = net['id']
					if id in seen:
						assert seen[id] == net, "inconsistent definitions for network %s" % (id,)
					else:
						seen[id] = net

		check_disks()
		check_networks()
		return rv

	@cached_property
	def base_bootstrap_config(self):
		identity = run_output(nix_eval_cmd(self._eval_args(state=False, attrib='info.bootstrap')))
		identity = xml2py(identity, self.cwd, file_digests=True)

	def bootstrap_image(self, size):
		provider = getpath(self.nix_expr, ['provider','type'])
		# we build our own `identity` hash, which contains:
		# - the complete closure of config.bootstrap (a simple nix expr)
		# - the (parsed & normalized) syntax of ./nix/google-compute-image.nix
		# We specifically don't include the system (nixpkgs) dependencies, as
		# that would cause a lot of unnecessary rebuilds - we'll assume a compatible
		# image regardless of nixpkgs version
		identity = self.base_bootstrap_config
		import hashlib
		identity = repr((identity, size)) + "\n" + run_output([
			'nix-instantiate', '--parse',
			os.path.join(nixlib, "google-compute-image.nix")])
		# log.debug("bootstrap identity config:\n%s", identity)
		identity = hashlib.sha1(identity).hexdigest()[:12]
		log.debug("bootstrap identity digest: %s", identity)
		def build():
			dest_path = os.path.join(self.storage_dir, "bootstrap-%sg" % (size,))
			log.info("building bootstrap image %s\n  in %s\n  (this will take a while if it has not been previously built)", identity, dest_path)
			run_cmd(nix_build_cmd(dest_path,
				self._eval_args(state=False, attrib='bootstrap.'+provider+'.image')
				+ ['--arg', 'diskSize', "%d" % size]
			))
			return os.path.join(dest_path, 'nixos.tar.gz')
		return BootstrapImage(identity, build)

	def build_machine_configs(self, dest):
		output = run_output(nix_build_cmd(dest, self._eval_args(state=True, attrib='config')))
		return output

	def dump_machine_config(self, attr, expr, state=True, strict=False):
		if attr and expr: raise AssertionError("both attr and expr given")
		cmd = ['nix-instantiate', '--eval'] + self._eval_args(state=state)
		if attr:
			cmd += ['-A', 'machines.'+attr]
		else:
			cmd += ['-A', 'evalFn', '--arg', 'fn', expr]
		if strict:
			cmd.append('--strict')
		cmd = nix_cmd(cmd)
		dump_cmd(cmd)
		proc = subprocess.Popen(cmd, stdout = subprocess.PIPE)
		outputSize = 0
		SILENCE = False
		while True:
			chunk = proc.stdout.readline(100)
			if not chunk: break
			if SILENCE: continue # just exhaust stdout
			outputSize += len(chunk)
			# err("CHUNK: %d" %(len(chunk)))
			sys.stdout.write(chunk)
			if outputSize > DUMP_SIZE_LIMIT:
				# nix-instantiate needs a kill -9, which is not very friendly
				err("\n\n  ( ... )\n  output too large; cancelling to prevent infinite recursion\n")
				SILENCE = True
				proc.kill()
		code = proc.wait()
		if code != 0:
			raise AssertionError("command failed with status %d" % code)

	@cached_property
	def hook_names(self):
		output = run_output(nix_eval_cmd(self._eval_args(state=False, attrib='customHookNames')))
		rv = xml2py(output, self.cwd)
		return rv

	def _eval_args(self, state, attrib=None):
		rv= [
			"--arg", "checkConfigurationOptions", "false",
			"--argstr", "networkConfig", self.path,
		]
		if state:
			rv.extend(["--arg", "networkState", self.state_path])
		if attrib is not None:
			rv.extend(['-A',attrib])
		rv.append(os.path.join(nixlib, "eval-machine-info.nix"))
		return rv

	@cached_property
	def hooks(self):
		output = run_output(nix_eval_cmd(self._eval_args(state=True, attrib='hooks')))
		# err(output)
		# json_str = json.loads(output)
		# err(repr(json_str))
		# assert isinstance(json_str, unicode)
		# rv = json.loads(json_str)
		# err(repr(rv))
		rv = xml2py(output, self.cwd)
		# log.debug("hooks: %r", rv)
		return rv
	
	def run_hook(self, hooks, machine):
		if not hooks: return
		#XXX defaults from options.nix don't seem to be present?
		for script in hooks:
			script_name = script.get('name', '(unnamed)')
			def summarize(action):
				log.info("%s script on %s: %s", action, machine.id, script_name)
				log.debug("Script definition: %r", script)

			if script.get('skip') is True:
				summarize('Skipping')
				continue
			summarize('Running')

			runner = script.get('runner', 'bash -eu').split(' ') # XXX do we ever need spaces?
			# XXX if `runner` is a nix store path (and `local` is False),
			# make sure it's present on remote
			# XXX allow derivations?
			args = script.get('args', [])
			contents = script['script']

			sudo = script.get('sudo', False)

			if script.get('local', False):
				# easy, just run it:
				sudo_arg = ['sudo'] if sudo else []
				with tempfile.NamedTemporaryFile() as f:
					f.write(contents)
					f.flush()
					run_cmd(sudo_arg + runner + [f.name] + args, cwd=self.cwd)
			else:
				# trickier, we need to run it via SSH
				assert '__MAPNIX_EOF' not in contents
				bash_script='''
					set -eu
					fname="/tmp/mapnix-$$"
					trap "rm -rf $fname" EXIT
					cat > $fname <<"__MAPNIX_EOF"\n'''+contents+'''\n__MAPNIX_EOF
					exec ''' + ' '.join(runner) + ''' $fname "$@"
				'''
				cmd = 'bash -euc ' + pipes.quote(bash_script) + ' -- ' + ' '.join(map(pipes.quote, args))
				if sudo:
					cmd = 'sudo ' + cmd
				# log.debug("RUNNING: %s", cmd)
				machine.run_ssh(cmd)

	@cached_property
	def machines(self):
		# err(repr(self.state))
		cached_machines = self.state['machines']
		machine_configs = self.nix_expr['machines']
		ids = machine_configs.keys()
		assert set(ids) == set(cached_machines.keys()), "stale cache"

		return [
			Machine(id, cached_machines[id], machine_configs[id])
			for id in ids]

	def each_machine_with_hooks(self, action, custom=False, bracket=True, limit=None, fn=None):
		assert limit is not None, "`limit` argument missing"
		hooks = self.hooks
		found_one = False
		if custom:
			bracket = False
		items = []

		for machine in self.machines:
			# XXX check tags, too
			if limit and machine.id not in limit:
				log.debug("Skipping machine %s", machine.id)
				continue
			hooks = self.hooks.get(machine.id, None)
			if hooks is None:
				hooks = DEFAULT_HOOK
			else:
				if custom:
					hook = getpath(hooks, ['custom', action], DEFAULT_HOOK)
					if hook is DEFAULT_HOOK:
						log.info("%s has no such hook: %s", machine, action)
				else:
					hook = hooks.get(action, DEFAULT_HOOK)

			if hook is not DEFAULT_HOOK:
				found_one = True

			items.append((machine, hook))

		if custom and not found_one:
			raise AssertionError("No such hook: %s" %(action,))

		# note: the above loop is fully evaluated in order to catch
		# configuration errors before we start running anything

		if fn is None: fn = lambda machine: None
		def loop(pair):
			machine, hook = pair
			if not bracket:
				# no point in having before / after split for a custom hook:
				log.debug("Running %s hook on %s", action, machine.id)
				self.run_hook(hook.get('actions', []), machine)
				fn(machine)
			else:
				log.debug("Running %s[before] hook on %s", action, machine.id)
				self.run_hook(hook.get('before', []), machine)
				fn(machine)
				log.debug("Running %s[after] hook on %s", action, machine.id)
				self.run_hook(hook.get('after', []), machine)


		# split machines into batches
		groups = groupby(items, lambda i: i[1]['order']['batch'])
		# sort groups by key
		groups = sorted(groups, key = lambda g: g[0])
		# sort the machines in each group, and drop group key
		groups = [sorted(group[1], key=lambda item: item[1]['order']['sort']) for group in groups]

		log.debug("grouped and ordered machines: %r", [[machine.id for (machine, hook) in group] for group in groups])

		# validate that `parallel` is consistent within each batch
		for items in groups:
			parallel = set([hook['order']['parallel'] for machine, hook in items])
			if len(parallel) > 1:
				machine_ids = [machine.id for machine,hook in items]
				raise AssertionError("conflicting `parallel` values for %r across machines %r: %s" %(action, machine_ids, '/'.join(map(str, parallel))))
			assert parallel.pop() is not None

		# note: run_batch_action is partitioned on groups - so if the first group fails,
		# further groups will not be attempted. Should this be configurable?
		for items in groups:
			parallel = items[0][1]['order']['parallel']
			run_batch_action(loop, items, parallel=parallel)

	def provider(self):
		attrs = self.nix_expr['provider']
		return PROVIDERS[attrs['type']](attrs, self)

class Diff(object):
	def __init__(self):
		self._nodes = []
	
	def add(self, n):
		# allow arbitrarily-nested diff lists,
		# because conceptual diffs (e.g recreate machine)
		# are composed of individual diffs with different `order`s
		if isinstance(n, list):
			for item in n:
				self.add(item)
		else: self._nodes.append(n)
	
	@property
	def nodes(self):
		return sorted(self._nodes, key=lambda d: d.order)

class DiffNode(object):
	ok = None
	def print(self):
		print(" - " + str(self))

class ResourceSet(object):
	def __init__(self):
		self.used = {}
		self.unused = {}
	
	def add(self, key, value):
		assert key not in self.unused
		self.unused[key] = value
		return value

	def use(self, key, allow_duplicate=False, allow_missing = False):
		if key in self.used:
			if allow_duplicate:
				return self.used[key]
			assert False, "resource %s claimed multiple times" % (key,)
		try:
			node = self.unused.pop(key)
		except KeyError:
			if allow_missing:
				node = None
			else:
				raise
		self.used[key] = node
		return node

	def create_dummy(self, key):
		# certain resources are created if they don't exist in `used`, but
		# we don't want to do that multiple times for the one key
		self.used[key] = None
	
	def _iter_each(self, method):
		for collection in (self.used, self.unused):
			# err("Calling %s on %r" %(method, collection))
			for x in getattr(collection, method)():
				yield x

	def items(self): return self._iter_each('items')
	def keys(self): return self._iter_each('keys')
	def values(self): return self._iter_each('values')
	
	def __getitem__(self, key):
		try:
			return self.used[key]
		except KeyError:
			return self.unused[key]

	def __contains__(self, key):
		return key in self.used or key in self.unused


class DiffChange(DiffNode):
	ok = False
	def __init__(self, order, desc, fix, confirm = False):
		self.order = order
		self.desc = desc
		self.fix = fix
		self.confirm = confirm

	def __str__(self):
		return self.desc

# make aliases for each Ordering on DiffChange:
def add_diffchanged_orderings():
	for key in dir(Order):
		if key.startswith('_'): continue
		def build(order):
			def cons(cls, *a, **k):
				return cls(order, *a, **k)
			return classmethod(cons)
		setattr(DiffChange, key, build(getattr(Order, key)))
add_diffchanged_orderings()

class DiffEq(DiffNode):
	order = Order.Noop
	ok = True
	def __init__(self, desc):
		self.desc = desc
	def __str__(self):
		return self.desc

class Vm(object):
	pass

class ProviderCommon(object):
	project = None

	def sshCommand(self, machine):
		cmd = ['ssh']
		host = machine['targetHost']
		if 'sshUser' in self.project:
			host += '@'+self.project['sshUser']
		if 'sshPort' in machine:
			cmd.extend(['-p',str(self.project['sshPort'])])
		return cmd

	def disk_id(self, disk, machine_id):
		id = disk['id']
		if id == 'root':
			# root image IDs are just named after their machine
			id = machine_id # + '$' + id
		return id


class Libvirtd(ProviderCommon):
	def __init__(self, project, deployment):
		self.project = project.copy()
		self.deployment = deployment
		self.project_id = key_safe(self.project['id'])
		self.resource_dir = deployment.storage_dir

		try:
			os.makedirs(self.disk_path(None))
		except (IOError, OSError) as e:
			if e.errno != errno.EEXIST: raise
	
	def disk_path(self, disk, machine_id=None):
		base = os.path.join(self.resource_dir, 'disks')
		if disk is None:
			return base

		return os.path.join(base, self.disk_id(disk, machine_id))

	def root_disk_path(self, machine_id):
		return self.disk_path({'id':ROOT_DISK_ID}, machine_id)
	
	def new_root_disk(self, disk, machine_id):
		bootstrap_image = os.path.join(self.disk_path(None), 'root')
		def create():
			# XXX locking?
			dest = self.disk_path(disk, machine_id)
			with atomic_write(dest) as dest:
				if not os.path.exists(bootstrap_image):
					with atomic_write(bootstrap_image) as bootstrap_tmp:
						run_cmd([
							"nix-build",
							"-o", bootstrap_tmp,
							"--arg", "checkConfigurationOptions", "false",
							"--show-trace",
							"--argstr", "networkConfig", self.deployment.path,
							"--argstr", "size", str(disk['size']),
							"-A", "bootstrap.libvirtd",
							os.path.join(nixlib, "eval-machine-info.nix"),
						])
				shutil.copyfile(os.path.join(bootstrap_image, 'disk.qcow2'), dest)
		return DiffChange.CreateDisk("create bootstrap disk " + disk['id'] + ' for ' + machine_id, create)

	def new_instance(self, name, guid, machine):
		def create():
			machine_xml = self._make_domain_xml(
				id=guid,
				disk=self.root_disk_path(name),
				machine = machine
			)
			xml_path = os.path.join(self.resource_dir, 'domain-'+name+'.xml')
			with open(xml_path, 'w') as f:
				f.write(machine_xml)
			log.debug("making machine from xml:\n" + machine_xml)
			# run_cmd(self.virsh_cmd + ["create", xml_path]);
			run_cmd(['sudo','virsh', '-c','qemu:///system'] + ["create", xml_path]);
		return DiffChange.CreateMachine("create machine %s" % name, create)

	def new_disk(self, disk):
		def create():
			dest = self.disk_path(disk, None)
			with atomic_write(dest) as dest:
				# TODO: run via nix?
				run_cmd(['qemu-img', 'create', '-f', 'qcow2', dest, "%dG" % disk['size']])
		return DiffChange(Order.CreateDisk, "create empty disk " + disk['id'], create)
	
	@property
	def connection_str(self):
		return self.project.get('connection', 'qemu:///session')

	@property
	def virsh_cmd(self):
		return ['virsh', '-c', self.connection_str]

	def machine_id(self, name):
		return key_safe(name)
	def machine_guid(self, name):
		return self.project_id + '$' + key_safe(name)

	def extant_machines(self):
		lines = run_output(self.virsh_cmd + ['list', '--name']).splitlines()
		return list(filter(None, lines))

	def extant_disks(self):
		rv = {}
		for fname in os.listdir(self.disk_path(None)):
			# id = os.path.splitext(fname)[0]
			id = fname
			assert id not in rv
			rv[id] = fname
		return rv

	def _make_domain_xml(self, id, disk, machine):
		# err(repr(machine['hardware']['memory']))
		def iface(n):
			return "\n".join([
				'    <interface type="network">',
				'      <source network="{0}"/>',
				'    </interface>',
			]).format(n)

		# XXX expose this in deploy.networks?
		networks = ['default'];

		domain_fmt = "\n".join([
			'<domain type="kvm">',
			'  <name>{name}</name>',
			'  <memory unit="MiB">{mem:d}</memory>',
			'  <vcpu>1</vcpu>',
			'  <os>',
			'    <type arch="x86_64">hvm</type>',
			'  </os>',
			'  <devices>',
			'    <emulator>{emu}</emulator>',
			'    <disk type="file" device="disk">',
			'      <driver name="qemu" type="qcow2"/>',
			'      <source file="{diskImg}"/>',
			'      <target dev="hda"/>',
			'    </disk>',
			# '    <interface type="user">',
			# '      <source network="default"/>',
			# '    </interface>',
			"\n".join(list(map(iface, networks))),
			'    <graphics type="sdl" display=":0.0"/>',
			# '    <input type="keyboard" bus="usb"/>',
			'    <input type="mouse" bus="usb"/>',
			'  </devices>',
			'</domain>',
		])

		qemu_kvm = spawn.find_executable("qemu-kvm")
		assert qemu_kvm != None

		return domain_fmt.format(
			name = id,
			emu = qemu_kvm,
			diskImg = disk,
			mem = machine['hardware']['memory']
		)

	def diff(self, _opts):
		ignore(_opts)
		rv = Diff()

		disks = self.extant_disks()
		log.debug("current disks: %r", disks)

		machines = self.extant_machines()
		log.debug("current machines: %r", machines)
		for name, machine_def in self.deployment.nix_expr['machines'].items():
			machine_name = self.machine_id(name)
			machine_guid = self.machine_guid(name)
			log.debug("checking machine: %s", name)

			for disk in machine_def['disks']:
				id = self.disk_id(disk, machine_name)
				is_root = disk['id'] == ROOT_DISK_ID
				if id in disks:
					if is_root:
						diff = DiffEq("root disk (%s)" % (machine_name,))
					else:
						diff = DiffEq("TODO: check config of " + id)
					rv.add(diff)
				else:
					if is_root:
						rv.add(self.new_root_disk(disk, machine_name))
					else:
						rv.add(self.new_disk(disk))

			if machine_guid in machines:
				# machine_xml = run_output(self.virsh_cmd + ['dumpxml'])
				diff = DiffEq("TODO: check config of " + machine_name)
			else:
				diff = self.new_instance(name, machine_guid, machine_def)
			rv.add(diff)

		return rv

class Adhoc(ProviderCommon):
	def __init__(self, project, deployment):
		self.project = project.copy()
		self.deployment = deployment

	def diff(self, _opts):
		ignore(_opts)
		# adhoc doesn't do any infrastructure management, but it does
		# iterate through all the machine definitions to populate
		# deployment.state
		rv = Diff()
		for name, machine in self.deployment.nix_expr['machines'].items():
			address = machine['ssh']['host']
			self.deployment.add_state(
				['machines', name, 'publicAddress'], address)
			rv.add(DiffEq("machine %s (%s)" % (name, address)))
		return rv

class Gce(ProviderCommon):
	def __init__(self, project, deployment):
		self.project = project.copy()
		self.deployment = deployment
		self._apis = {}

	def get_region(self, zone):
		return zone.rsplit('-',1)[0]

	def api(self, zone=None, machine=None):
		if zone is None:
			if machine is not None:
				zone = getpath(self.deployment.nix_expr, ['machines', machine, 'gce', 'zone'])

		if zone not in self._apis:
			project = self.project['project']
			from libcloud.compute.types import Provider
			from libcloud.compute.providers import get_driver
			log.debug("Creating GCE driver for %s", zone)
			driver = get_driver(Provider.GCE)
			self._apis[zone] = driver(
					self.project['account'], os.path.join(self.deployment.cwd, self.project['keyFile']),
					datacenter = zone, project = project)
			if zone is not None:
				assert self._apis[zone].zone, "invalid zone: %s" % zone
		return self._apis[zone]
	
	@cached_property
	def storage_api(self):
		from libcloud.storage.types import Provider
		from libcloud.storage.providers import get_driver
		log.debug("Creating GCE storage driver")

		driver = get_driver(Provider.GOOGLE_STORAGE)
		if 'storage' not in self.project:
			raise AssertionError("storage API credentials not set.\n"
					"Note that these are not the same as your GCE credentials, and require you to "
					"enable \"interoperable access\" for the storage project you'll be using")

		storage_creds = self.project['storage']
		with open(os.path.join(self.deployment.cwd, storage_creds['secretFile'])) as f:
			secret = f.read().strip()
		return driver(storage_creds['key'], secret)

	def record_machine(self, machine_id, address):
		self.deployment.add_state(['machines', machine_id, 'publicAddress'], address)

	def record_disk(self, machine_id, disk, info):
		path = "/dev/disk/by-id/{type}-0Google_PersistentDisk_{name}".format(
			type=info['interface'].lower(),
			name = info['deviceName'])
		self.deployment.add_state(
			['machines', machine_id, 'disks', disk['id'], "devicePath"],
			path)

	def _populate_resources(self, opts):
		res = GceResources()

		active_zones = set()
		active_zones.update(self.project.get('zones',[]))
		for machine_id, machine in self.deployment.nix_expr['machines'].items():
			active_zones.add(getpath(machine, ['gce','zone']))
		active_regions = set(map(self.get_region, active_zones))

		for region in active_regions:
			log.debug("checking region %s", region)
			res.disks[region] = ResourceSet()
			res.addresses[region] = ResourceSet()
			# XXX this is dumb. gce requires a zone, even for region-wide operations.
			# So we'll just pick the first active zone in this region
			zone_matches_region = lambda zone: zone.startswith(region + '-')
			first_active_zone = next(iter(filter(zone_matches_region, active_zones)))
			api = self.api(zone=first_active_zone)

			for disk in api.list_volumes():
				log.debug("saw disk: %s", disk.name)
				res.disks[region].add(disk.name, disk)

			for address in api.ex_list_addresses():
				log.debug("saw address %s - %s", address.name, address.address)
				res.addresses[region].add(address.name, address)

		for zone in active_zones:
			log.debug("checking zone %s", zone)
			# populate all resources in this zone
			res.instances[zone] = ResourceSet()
			api = self.api(zone=zone)
			for node in api.list_nodes():
				log.debug("saw machine %s", node.name)
				res.instances[zone].add(node.name, node)

				for disk in node.extra['disks']:
					# XXX this will bail on multiple attached (readonly disks)
					log.debug("saw volume %s attached to %s", disk['deviceName'], node.name)
					res.disk_assignment.add(disk['deviceName'], (node, disk))

		log.debug("checking global resources")
		api = self.api()
		if not opts.assume_network_ok:
			for network in api.ex_list_networks():
				log.debug("saw network: %s", network.name)
				res.networks.add(network.name, network)
				res.firewalls[network.name] = ResourceSet()

			for firewall in api.ex_list_firewalls():
				netname = firewall.network.name
				res.firewalls[netname].add(firewall.name, firewall)
				log.debug("saw firewall rule %s", firewall.name)

		for image in api.list_images():
			res.images.add(image.name, image)

		storage_api = self.storage_api
		bucket = self.get_bucket(self.bootstrap_bucket_name)
		if bucket:
			for blob in storage_api.iterate_container_objects(bucket):
				res.bootstrap_store_files.add(blob.name, blob)

		return res

	def diff(self, opts):
		from libcloud.common.google import ResourceNotFoundError
		rv = Diff()
		res = self._populate_resources(opts)

		# then we go through the _wanted_ resources and mark them as used,
		# diffing any sub-resources as we go
		for machine_id, machine in self.deployment.nix_expr['machines'].items():
			gce = machine['gce']
			zone = gce['zone']
			# pdb_trace()
			region = self.get_region(zone)
			api = self.api(zone=zone)

			node = None
			try:
				node = res.instances[zone].use(machine_id)
			except KeyError:
				rv.add(self.missing_machine(machine_id, gce))
			else:
				# check mismatches first, as we won't bother performing certain checks if
				# the node needs recreating
				attached_networks = set([net['network'].rsplit('/',1)[1] for net in node.extra['networkInterfaces']])
				log.debug("Attached networks are %r" , attached_networks)
				attached_network = attached_networks.pop() if len(attached_networks) == 1 else None

				if node.size != gce['instanceType']:
					log.debug("node is the wrong instance type want %s, but it is %s", gce['instanceType'], node.size)
					rv.add(self.outdated_machine(machine_id, node, gce))
					node = None
				elif attached_network != gce['network']['id']:
					rv.add(self.outdated_machine(machine_id, node, gce))
					node = None

			if node:
				# check tags
				expected_tags = set(gce.get('tags', []))
				current_tags = set(node.extra.get('tags',[]))
				if expected_tags != current_tags:
					rv.add(DiffChange.Modify("update %s tags" % (machine_id,), TODO))

			# check IP
			address_name = gce.get('staticAddress')
			address = None

			if address_name is None:
				if node:
					address = node.public_ips[0]
			else:
				try:
					found_address = res.addresses[region].use(address_name)
				except KeyError:
					rv.add(DiffChange.CreateAddress("new static address %s" %(address_name,), TODO))
				else:
					# this node _will_ be found at this IP, once created
					address = found_address.address
					if node and (found_address.address not in node.public_ips):
						rv.add(DiffChange.AttachAddress(
							"assign static address %s to %s" %(address_name, machine_id), TODO))

			if address is not None:
				self.record_machine(machine_id, address)
				if node:
					rv.add(DiffEq("machine %s (%s)" % (machine_id, address)))

			# else the above `new static address` or `create machine` change will call self.record_machine upon creation

			# check associated disks
			for disk in gce.get('disks', []):
				boot = disk['id'] == ROOT_DISK_ID

				bootstrap_image = None

				# check its bootstrap image / storage
				# Note that an image will only be needed if the disk does not yet exist.
				# But we want to keep images around if any disk is based on them, since
				# it's likely they'll be needed again in the future.
				# So we blindly mark any references image / bootstrap file as used,
				# regardless of whether we need them right now.

				image_info = disk.get('image')
				if image_info:
					# we'll only use this if the disk is missing, but we don't want
					# to delete it simply because the disk is already present. So mark
					# it as used
					image_id = image_info['id']
					res.images.use(image_id, allow_duplicate=True, allow_missing=True)
					storage = image_info.get('storage')
					if storage is not None:
						# just in case image_storage is in our bootstrap bucket, mark it as used too:
						filename = storage.rsplit('/',1)[1]
						res.bootstrap_store_files.use(filename, allow_duplicate=True, allow_missing=True)

				if boot:
					bootstrap_image = self.deployment.bootstrap_image(disk['size'])
					log.debug("machine %s uses bootstrap image %s" % (machine_id, bootstrap_image.filename))

					res.images.use(bootstrap_image.name, allow_duplicate=True, allow_missing = True)

					if bootstrap_image.filename in res.bootstrap_store_files.unused:
						rv.add(DiffEq("bootstrap image %s" % bootstrap_image.filename))
					res.bootstrap_store_files.use(bootstrap_image.filename, allow_duplicate=True, allow_missing = True)

				disk_id = self.disk_id(disk, machine_id)
				try:
					disk_node = res.disks[region].use(disk_id)
				except KeyError:
					disk_image = None
					if bootstrap_image is not None:
						# we've already marked bootstrap_image as used above,
						# so just check for its presence

						log.debug("have bootstrap image %s, it is %r", bootstrap_image.name, res.images.used[bootstrap_image.name])
						if res.images.used[bootstrap_image.name] is None:
							# bootstrap image not created.
							rv.add(self.missing_image(api,
								name=bootstrap_image.name,
								storage_url=self.storage_url(
									bucket = self.bootstrap_bucket_name,
									filename = bootstrap_image.filename)))

							if res.bootstrap_store_files.used[bootstrap_image.filename] is None:
								# bootstrap storage not created
								rv.add(self.missing_bootstrap_storage(bootstrap_image))

						disk_image = bootstrap_image.name
					else:
						# not a boot image. Check for hardcoded image:
						image_info = image_id = disk.get('image')
						if image_info:
							# storage_api = self.storage_api
							disk_image = image_info['id']
							try:
								api.get_image(disk_image)
							except ResourceNotFoundError:
								disk_storage = image_info.get('storage')
								rv.add(self.missing_image(api, name=disk_image, storage_url=disk_storage))

					rv.add(self.missing_disk(api, disk_id, disk, image_id=disk_image))
					if not boot:
						rv.add(self.missing_mount(machine_id=machine_id, disk=disk))
				else:
					# disk exists. check its attachment
					try:
						(assigned_node, assignment) = res.disk_assignment.use(disk_id)
					except KeyError:
						# not attached to anything
						if node is not None:
							# only do attachments when machine exists
							# (in missing_machine, we'll mount any disks as needed)
							rv.add(self.missing_mount(machine_id=machine_id, disk=disk))
					else:
						if assigned_node.name == machine_id:
							if node:
								rv.add(DiffEq("volume %s attached to %s" % (disk_id, machine_id)))
							self.record_disk(machine_id, disk, assignment)
						else:
							# attached to the wrong node!
							# XXX we can't tell if it's attached as a boot disk,
							# in which case we won't be able to detach it
							rv.add([
								self.unwanted_mount(assigned_node, disk_node),
								self.missing_mount(machine_id=machine_id, disk=disk),
							])

			# check its network
			if not opts.assume_network_ok:
				net = gce['network']
				net_id = net['id']
				def firewall_spec(fw):
					# return a nix-equivalent spec from an existing firewall rule
					rv = {}
					if fw.source_tags: rv['sourceTags'] = fw.source_tags
					if fw.target_tags: rv['targetTags'] = fw.target_tags
					if fw.source_ranges != ['0.0.0.0/0']: rv['sourceRanges'] = fw.source_ranges
					for rule in fw.allowed:
						proto = rule['IPProtocol']
						if proto not in rv:
							if proto == 'icmp':
								rv[proto] = True
								continue
							rv[proto] = []
						for port in rule['ports']:
							if '-' not in port:
								port = int(port)
							rv[proto].append(port)
					return rv

				if net_id not in res.networks.used:
					# networks are multiple-use; just check them the first time:
					net_node = res.networks.use(net_id, allow_missing = None)
					if net_node is None:
						rv.add(self.missing_network(api, net))
					else:
						ok = True
						if net['cidr'] != net_node.cidr:
							log.error("cidr has changed, this is not supported")
							rv.add(DiffChange.Modify("change cidr of network %s" %(net_id,), NOT_IMPLEMENTED))

						firewall_rules = res.firewalls.get(net_id)
						for rule_name, rule in net.get('rules', {}).items():
							try:
								existing_rule = firewall_rules.use(self.firewall_id(net_id, rule_name))
							except KeyError:
								ok = False
								rv.add(self.missing_firewall(api, net_node, rule_name, rule))
							else:
								spec = firewall_spec(existing_rule)
								if spec != rule:
									ok = False
									log.debug("existing firewall %r differs from %r; recreating", spec, rule)
									rv.add([
										self.unwanted_firewall(existing_rule),
										self.missing_firewall(api, net_node, rule_name, rule)
									])
								# we don't bother to report correct rules, as that's noisy

						for rule_name, rule in firewall_rules.unused.items():
							ok = False
							rv.add(self.unwanted_firewall(rule))

						if ok: rv.add(DiffEq("network %s" %(net_id)))
						# if not ok, we'll have added individual diffs for the differing components

		res.cleanup_unwanted(self, rv, opts)
		return rv

	def create_firewall(self, api, network, name, rule):
		rules = [];
		for proto in ('tcp','udp'):
			if proto in rule:
				rules.append({ 'IPProtocol': proto, 'ports': list(map(str, rule[proto])) })
		if 'icmp' in rule:
			rules.append({ 'IPProtocol': 'icmp' })

		api.ex_create_firewall(
			name=self.firewall_id(network.name, name),
			allowed=rules,
			network=network,
			source_ranges = rule.get('sourceRanges'),
			source_tags = rule.get('sourceTags'),
			target_tags = rule.get('targetTags')
		)

	def firewall_id(self, net_id, name):
		return net_id + "-" + name

	def missing_network(self, api, network):
		def make():
			# create network
			# then add each firewall rule
			created = api.ex_create_network(
				name=network['id'],
				cidr=network['cidr'])
			for rule_name, rule in network.get('rules', {}).items():
				self.create_firewall(api, created, rule_name, rule)

		return DiffChange.CreateNetwork(
			"create network %s" %(network['id']),
			make)

	def missing_firewall(self, api, network, rule_name, rule):
		id = self.firewall_id(network.name, rule_name)
		return DiffChange.CreateFirewall(
			"create firewall rule %s" %(id),
			lambda: self.create_firewall(api, network, rule_name, rule))

	def unwanted_image(self, img):
		def rm():
			assert img.delete()
		return DiffChange.DeleteImage("remove image %s"%(img.name), rm)

	def unwanted_store_file(self, obj):
		# XXX ideally this wouldn't need confirmation, but
		# GCE has distinct compute & storage credentials, and only one storage driver
		# can be the default. This means you'll probably use the one storage account for
		# multiple deployments, so we can't tell if the image is actually unused
		def rm():
			assert self.storage_api.delete_object(obj)
		return DiffChange.DeleteImageStorage(
			"remove store file %s"%(obj.name),
			rm, confirm=True)

	def unwanted_firewall(self, rule):
		return DiffChange.DeleteFirewall(
			"remove firewall rule %s"%(rule.name),
			lambda: destroy(rule))

	def unwanted_address(self, node):
		return DiffChange.DeleteAddress(
			"remove static address %s"%(node.name),
			lambda: destroy(node), confirm=True)

	def unwanted_machine(self, node):
		return DiffChange.DeleteMachine(
			"destroy machine %s" % (node.name,),
			lambda: destroy(node))

	def missing_machine(self, id, spec):
		def make():
			ignore(id, spec)
			api = self.api(zone=spec['zone'])
			disk = self.get_disk(api, id)
			record_address = False
			static_address = spec.get('staticAddress')
			if static_address:
				address = self.get_address(api, spec['staticAddress'])
			else:
				record_address = True
				address = 'ephemeral'

			# libcloud requires an image-like object, but
			# gce doesn't care (if the boot disk already exists)
			image = Object()
			image.name = "FAKE"

			node = api.create_node(name=id,
				size=spec['instanceType'],
				image=image,
				location=spec['zone'],
				ex_network=getpath(spec, ['network','id'], 'default'),
				ex_tags = spec.get('tags'),
				ex_boot_disk=disk,
				use_existing_disk=True,
				external_ip = address,
				ex_disk_auto_delete = False,
			)
			if record_address:
				self.record_machine(id, node.public_ips[0])

		rv = [
			DiffChange.CreateMachine("new machine %s" % (id,), make),
		]
		# most resources are associated upon creation
		# (e.g network, boot disk, address),
		# but non-boot disks need explicit attachment
		for disk in spec.get('disks', []):
			disk_name = disk['id']
			if disk_name == ROOT_DISK_ID: continue
			rv.append(self.missing_mount(machine_id=id, disk=disk))
		return rv

	def outdated_machine(self, id, node, spec):
		return [
			self.unwanted_machine(node),
			self.missing_machine(id, spec),
		]

	def unwanted_mount(self, instance, disk):
		def rm():
			assert instance.driver.detach_volume(disk, instance)
		return DiffChange.DetachDisk(
			"detach disk %s from %s" % (disk.name, instance.name), rm)

	def missing_mount(self, machine_id, disk):
		disk_id = self.disk_id(disk, machine_id)
		def mount():
			api = self.api(machine=machine_id)
			instance = self.get_instance(api, machine_id)
			disk_node = self.get_disk(api, disk_id)
			assert api.attach_volume(instance, disk_node)
			for assignment in self.get_instance(api, machine_id).extra['disks']:
				if assignment['deviceName'] == disk_id:
					self.record_disk(machine_id, disk, assignment)
					break
			else:
				assert False, "Couldn't find disk %s" % disk_id

		return DiffChange.AttachDisk(
			"attach disk %s to %s" % (disk_id, machine_id), mount)

	def missing_disk(self, api, disk_id, disk, image_id):
		common_kw = {
			'name': disk_id,
			'use_existing': False,
		}
		if image_id is None:
			# blank disk
			def build():
				api.create_volume(size=disk['size'], **common_kw)
			return DiffChange.CreateDisk(
				"new unformatted volume %s" % (disk_id,),
				build)

		else:
			# disk from existing image
			def build():
				api.create_volume(size=None, image = image_id, **common_kw)
			return DiffChange.CreateDisk(
				"new volume %s (from image %s)" % (disk_id,image_id),
				build)

	def parse_bucket_url(self, url):
		match = re.match('^gs://([^/]+)/?$', url)
		assert match, "Can't parse gs:// bucket URL: %s" % url
		return match.group(1)
	
	def storage_url(self, bucket, filename):
		return 'gs://%s/%s' % (bucket, filename)

	@property
	def bootstrap_bucket_name(self):
		return self.parse_bucket_url(
			getpath(self.deployment.nix_expr, ['bootstrap', 'storagePrefix']))

	def get_bucket(self, name):
		return self.storage_api.get_container(container_name=name)

	def get_bucket_opt(self, name):
		from libcloud.storage.types import ContainerDoesNotExistError
		try:
			return self.get_bucket(name)
		except ContainerDoesNotExistError:
			return None

	def missing_image(self, api, name, storage_url):
		if storage_url.startswith('gs://'):
			storage_url = 'https://storage.googleapis.com/' + storage_url[5:]

		def make():
			api.ex_create_image(name=name, volume=storage_url, use_existing = False)

		return DiffChange.CreateImage("new image %s" %(name,), make)

	def missing_bootstrap_storage(self, img):
		bucket_name = self.bootstrap_bucket_name

		def make():
			# from libcloud.utils import files as fileutils
			api = self.storage_api
			path = img.build()

			container = self.get_bucket_opt(bucket_name)
			if not container:
				log.debug("creating storage bucket %s", bucket_name)
				container = api.create_container(bucket_name)

			with open(path, 'rb') as infile:
				stat = os.fstat(infile.fileno())
				assert stat.st_size > 0
				size_mb = stat.st_size / 1000000

				# XXX if gcs supported multipart upload, this would be useful :/
				# def iterator():
				# 	processed = 0
				# 	processed_pct = 0
				# 	for chunk in fileutils.read_in_chunks(infile):
				# 		yield chunk
				# 		processed += len(chunk)
				# 		# print("processed: %d of %d bytes" % (processed, stat.st_size))
				# 		new_pct = processed / (stat.st_size * 100)
				# 		if new_pct > processed_pct:
				# 			print(" .. %d%%" %(new_pct))
				# 			processed_pct = new_pct

				log.info("Uploading %s image (%smb)", img.name, size_mb)
				api.upload_object_via_stream(
					iterator=infile,
					container=container,
					object_name=img.filename
				)

		return DiffChange.CreateImageStorage(
			"Upload image %s to %s" % (img.filename, bucket_name),
			make)
	
	def unwanted_disk(self, node):
		return DiffChange.DeleteDisk("destroy volume %s" % (node.name,), lambda: destroy(node))

	def unwanted_network(self, node, firewall_rules):
		def rm():
			for fw in firewall_rules:
				destroy(fw)
			destroy(node)
		return DiffChange.DeleteNetwork("destroy network %s" % (node.name,), rm)
	
	def get_instance(self, api, id):
		return api.ex_get_node(id)

	def get_address(self, api, id):
		return api.ex_get_address(id)

	def get_disk(self, api, id):
		return api.ex_get_volume(id)

class GceResources():
	def __init__(self):
		# per-zone:
		self.instances = {}

		# per-region:
		self.disks = {}
		self.addresses = {}

		# global:
		self.disk_assignment = ResourceSet()
		self.networks = ResourceSet()
		self.images = ResourceSet()
		self.bootstrap_store_files = ResourceSet()

		self.firewalls = {} # keyed on network

	def cleanup_unwanted(self, provider, diff, opts):
		# destroy resources that we've seen but not used:
		# disk attachments
		for disk_id, (instance, disk) in self.disk_assignment.unused.items():
			if disk_id == instance.name:
				# boot disk, can't be detached
				continue
			disk_node = self.disks[instance.driver.region.name][disk_id]
			diff.add(provider.unwanted_mount(instance, disk_node))

		# machines
		for _, collection in self.instances.items():
			for node in collection.unused.values():
				diff.add(provider.unwanted_machine(node))

		# disks
		for _, collection in self.disks.items():
			for node in collection.unused.values():
				diff.add(provider.unwanted_disk(node))

		# networks
		if not opts.assume_network_ok:
			for node in self.networks.unused.values():
				if node.name == 'default':
					# never GC the default network
					continue
				diff.add(provider.unwanted_network(node, self.firewalls[node.name].values()))
		# NOTE: firewalls don't need explicit cleanup, as they're fully owned by a network
		# (so either the network gets deleted, or the firewalls get diffed as part of the
		# network checking)

		# disk images
		for node in self.images.unused.values():
			if not node.name.startswith(BootstrapImage.name_prefix):
				log.debug("Ignoring non-mapnix image %s", node.name)
				continue
			diff.add(provider.unwanted_image(node))

		# static addresses
		for _, collection in self.addresses.items():
			for node in collection.unused.values():
				diff.add(provider.unwanted_address(node))

		# store files
		for node in self.bootstrap_store_files.unused.values():
			diff.add(provider.unwanted_store_file(node))


PROVIDERS['libvirtd'] = Libvirtd
PROVIDERS['adhoc'] = Adhoc
PROVIDERS['gce'] = Gce

# used when parallel=None
class UnboundedSemaphore(object):
	def __enter__(self): pass
	def __exit__(self, _a, _b, _c): pass

def run_batch_action(fn, nodes, parallel):
	# 0 is equivalent to null
	if parallel == 0: parallel = None
	log.debug("performing batch action on %d items, parallel = %s", len(nodes), parallel)
	failed = []
	def wrap_errors(fn, arg):
		try:
			fn(arg)
		except Exception as e:
			if not summarize_error(e):
				log.error("", exc_info=True)
			failed.append(1)

	if parallel == 1:
		run = lambda node: wrap_errors(fn, node)
		def wait(): pass
	else:
		threads = []
		if parallel is None:
			lock = UnboundedSemaphore()
		else:
			assert parallel > 0, parallel
			lock = threading.Semaphore(parallel)

		def wait():
			for t in threads: t.join()

		def run(node):
			def _run():
				with lock:
					wrap_errors(fn, node)

			t = threading.Thread(target=_run)
			t.daemon = True
			threads.append(t)
			t.start()

	for node in nodes:
		run(node)
	wait()
	if failed:
		raise AssertionError("%s subtasks failed" % (len(failed,)))

def run():
	p = optparse.OptionParser()
	p.add_option('-s', '--skip-infrastructure', action='store_true')
	p.add_option('--no-skip-infrastructure', action='store_false', dest='skip_infrastructure')
	p.add_option('--build', dest='build_only', action='store_true', help='just build system config, to check its validity (implies --skip-infrastructure)')
	p.add_option('--status', action='store_true', help='print a summary of infrastructure state')
	p.add_option('--infrastructure', dest='infrastructure_only', action='store_true', help='stop after setting up infrastructure')
	p.add_option('--always-activate', action='store_true', help='activate even unchanged configurations')
	p.add_option('--hook', help='run a custom hook (separate multiple hooks with a comma)')
	p.add_option('--hooks', action='store_true', help='list custom hooks')
	p.add_option('--show-trace', help='show nix traces', action='store_true')
	p.add_option('-f', '--force', help='don\'t prompt before changing infrastructure', action='store_true')
	p.add_option('-q', '--quiet', help='less logging', action='count', default=0)
	p.add_option('-v', '--verbose', help='more logging', action='count', default=0)

	p.add_option('--dump',        help='dump a config attribute (e.g <machineName>.networking.hostname)', dest='dump_attr')
	p.add_option('--dump-expr',   help='evaluate a nix expression (a function which can accept any of {machines, pkgs, lib})')
	p.add_option('--dump-raw',    help='(with --dump[-expr]): don\'t load network state / facts', action='store_true')
	p.add_option('--dump-strict', help='(with --dump[-expr]): strictly evaluate', action='store_true')

	# Enumerating network + firewalls can be slow, and rarely changes. So add
	# an option to skip it (without skipping all infrastructure)
	p.add_option('--assume-network-ok', action='store_true', help='don\'t check network infrastructure')

	# TODO: maybe make this part of config, rather than CLI args?
	p.add_option('--leave-uncertain', action='store_true', help='don\'t delete (or ask about) resources which could be needed later (e.g. static addresses, extra disk images)')
	p.add_option('--limit', default=[], action='append', help='limit activation to just $MACHINE (name or tag)')

	opts, args = p.parse_args()

	log_diff = opts.verbose - opts.quiet
	if log_diff < 0:
		log_level = logging.WARN
	elif log_diff == 0:
		log_level = logging.INFO
	elif log_diff >= 1:
		log_level = logging.DEBUG
	else: assert False, log_diff

	logging.basicConfig(level=log_level)

	if opts.build_only or opts.dump_attr or opts.dump_expr:
		if opts.skip_infrastructure is None:
			opts.skip_infrastructure = True

	nix_file, = args
	dep = Deployment(nix_file)

	global SHOW_TRACE
	SHOW_TRACE = bool(opts.show_trace)

	if opts.hooks:
		hooks = set()
		for names in dep.hook_names.values():
			hooks.update(names)
		for hook in sorted(hooks):
			print(hook)
		return
	
	storage_dir = dep.make_storage_dir()

	def process_infrastructure():
		if opts.skip_infrastructure:
			log.info("using last-known state of infrastructure")
			dep.load_cached_state()
		else:
			heading("Checking infrastructure")
			# print("PROVIDER:", dep.provider())
			diffs = dep.provider().diff(opts)
			if opts.status:
				not_ok = 0
				for diff in diffs.nodes:
					diff.print()
					if not diff.ok:
						not_ok += 1
				return not_ok
			else:
				changes = []
				for diff in diffs.nodes:
					if diff.ok:
						diff.print()
					else:
						changes.append(diff)

				if opts.leave_uncertain:
					skipped = [c for c in changes if c.confirm]
					changes = [c for c in changes if not c.confirm]
					if skipped:
						heading("Skipping uncertain changes:")
						for diff in skipped:
							diff.print()

				if changes:
					heading("Changes required:")
					for diff in changes:
						diff.print()

					step = False
					if not opts.force:
						response = prompt("Apply the above changes? (Y)es / (s)tep / (c)ancel: ", ['y','s','c',''])
						if response in ('', 'y'): pass
						elif response == 's': step = True
						elif response == 'c': cancel()
						else: assert False, "Unknown response: %r" % response

					for diff in changes:
						diff.print()
						if step or (diff.confirm and not opts.force):
							if diff.confirm:
								print("** action requires explicit confirmation **")
							response = prompt("OK? (Y)es / (s)kip / (c)ancel: ", ['y','s','c',''])
							if response in ('', 'y'): pass
							elif response == 's': continue
							elif response == 'c': cancel()
							else: assert False, "Unknown response: %r" % response
						diff.fix()

				fact_subjects = list(filter(lambda m: 'gatherFacts' in m.config, dep.machines))
				if fact_subjects:
					facts = {}
					heading("Gathering facts")
					# gather facts in parallel
					def gather(machine):
						facts[machine.id] = machine.gather_facts()
					run_batch_action(gather, fact_subjects, parallel=10)

					# save them in serial, just to prevent races
					for machine_id, val in facts.items():
						dep.add_state(['machines', machine_id, 'facts'], val)
				dep.save_state()

	if opts.dump_attr or opts.dump_expr:
		def do_dump(state):
			dep.dump_machine_config(
					attr=opts.dump_attr,
					expr=opts.dump_expr,
					state=state,
					strict=opts.dump_strict)

		if opts.dump_raw:
			do_dump(state=False)
		else:
			process_infrastructure()
			do_dump(state=True)
		return

	rv = process_infrastructure()
	if rv is not None: return rv
	if opts.infrastructure_only: return

	if opts.hook:
		hook_names = opts.hook.split(',')
		for name in hook_names:
			heading("Running hook: " + name)
			dep.each_machine_with_hooks(name, custom=True, limit=opts.limit)
	else:
		heading("Building system configuration")
		config_root = os.path.join(storage_dir, 'config')

		dep.each_machine_with_hooks('build', bracket=False, limit=opts.limit)
		dep.build_machine_configs(config_root)
		if opts.build_only:
			heading("Configuration built successfully")
			return
		# print("--> " + config_root)
		heading("Pushing config")

		def push(machine):
			heading("Pushing config to %s" % (machine.target))
			machine.copy_closure(os.path.join(config_root, machine.id), dep.cwd)
		dep.each_machine_with_hooks('push', limit=opts.limit, fn=push)

		def apply_config(machine):
			heading("applying config to %s" % (machine.target))
			machine.activate_config(os.path.join(config_root, machine.id), force=opts.always_activate)
		# unlike building & pushing, we want to _try_ activating all nodes even if
		# one of them fails.
		heading("Applying config")
		dep.each_machine_with_hooks('activate', limit=opts.limit, fn=apply_config)

def summarize_error(e):
	if isinstance(e, AssertionError):
		if SHOW_TRACE or not str(e): return False
		err("Asssertion error: %s" % e)
		return True
	elif isinstance(e, subprocess.CalledProcessError):
		if SHOW_TRACE: return False
		# pdb_trace()
		# err(repr(e))
		err("%s command failed" % (e.cmd[0],) )
		return True
	else:
		return False

def main():
	try:
		sys.exit(run())
	except Exception as e:
		if summarize_error(e):
			sys.exit(1)
		else:
			raise

if __name__ == '__main__':
	main()
