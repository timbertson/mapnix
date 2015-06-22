{ config, pkgs, ... }:

## NOTE: Never use `types.string`.
## It sounds right, but is a Bad Thing. Use` types.str` instead.

with pkgs.lib;

let

  cfg = config.deployment;
  optional = t: { default = null; type = types.nullOr t;};
  nested = name: opts: mkOptionType { inherit name; getSubOptions = prefix: opts; };
  function = mkOptionType { name = "function"; check = builtins.isFunction; };
in

{

  # imports =
  #   [ ./ec2.nix
  #     ./route53.nix
  #     ./virtualbox.nix
  #     ./ssh-tunnel.nix
  #     ./auto-raid0.nix
  #     ./auto-luks.nix
  #     ./keys.nix
  #     ./gce.nix
  #     ./hetzner.nix
  #     ./container.nix
  #     ./libvirtd.nix
  #   ];

  options = {
    deployment = {
      # disks = mkOption {
      #   default = [];
      #   type = types.listOf types.optionSet;

      #   options = {
      #     id = mkOption {
      #       default = "root";
      #       type = types.str;
      #     };
      #     label = mkOption {
      #       default = "nixos";
      #       type = types.str;
      #     };
      #     size = mkOption {
      #       type = types.int;
      #     };
      #   };
      # };

      state = mkOption { default = null; type = types.nullOr types.attrs; };

      targetHost = mkOption { default = null; type = types.nullOr types.str; };
      sshPort = mkOption { default = null; type = types.nullOr types.int; };
      sysconfDir = mkOption { default = null; type = types.nullOr types.str; };
      signStore = mkOption { default = false; type = types.bool; };

      moduleArgs = mkOption { default = null; type = types.nullOr types.attrs; };

      gce = let
        gceImage = nested "gce image" {
          id = mkOption { type = types.str; };
          storage = mkOption (nullOr types.str);
        };
        gceDisk = nested "gce disk" {
          id = mkOption { type = types.str; };
          size = mkOption { type = types.int; };
          # label = mkOption { type = types.str; default="nixos"; };
          image = mkOption { default = null; types = types.nullOr gceImage; };
        };
        # gceOptions = nested "gce instance" {
        #   zone = { type = types.str; };
        #   instanceType = mkOption { type = types.str; };
        #   disks = mkOption { default = []; type = types.listOf gceDisk; };
        # };
        portSpec = types.listOf (types.either types.str types.int);
        networkRules = nested "network rules" {
          tcp = optional portSpec;
          udp = optional portSpec;
          icmp = optional types.bool;
          sourceRanges = mkOption { type = types.listOf (types.either types.str types.int); default = [ "0.0.0.0/0" ]; };
        };
        gceNetwork = nested "gce network" {
          id = mkOption { type = types.str; };
          rules = mkOption (optional (types.attrsOf networkRules));
          cidr = mkOption { type = types.str; };
        };
      # in optional gceOptions;
      in mkOption {
        default = null; type = types.nullOr types.optionSet;
        options = {
          zone = mkOption { type = types.str; };
          instanceType = mkOption { type = types.str; };
          disks = mkOption { default = []; type = types.listOf gceDisk; };
          network = mkOption { type = gceNetwork; };
          tags = mkOption (optional (types.listOf types.str));
          staticAddress = mkOption (optional types.str);
        };
      };

      gatherFacts = mkOption { default = null; type = types.nullOr types.optionSet;
        options = {
          module = mkOption { type = types.either types.str types.path; };
          args = mkOption { type = types.nullOr types.attrs; };
          networkAddress = mkOption { type = types.bool; default = false; };
        };
      };

      ssh = {
        host = mkOption (optional types.str);
        user = mkOption (optional types.str);
      };

      hooks = let
        script = {
          script = mkOption { type = types.str; };
          runner = mkOption {
            default="${pkgs.bash}/bin/bash";
            type = types.str;
          };
          local = mkOption { type = types.bool; default = false; };
          register = mkOption (optional types.str);
          skip = mkOption (optional types.bool);
          name = mkOption (optional types.str);
        };
        order = types.either function (nested "order" {
          sort = mkOption (optional types.unspecified);
          batch = mkOption (optional types.unspecified);
          parallel = mkOption { type = types.int; default = 0; };
        });
        base = {
          order = mkOption (optional order);
        };
        hookType = base // {
          # XXX sub-options aren't working
          before = mkOption { type = types.listOf types.attrs; options = script; default = []; };
          after =  mkOption { type = types.listOf types.attrs; options = script; default = []; };
        };

        customHookType = base // {
          # XXX sub-options aren't working
          actions = mkOption { type = types.listOf types.attrs; options = script; default = []; };
        };
      in mkOption { default = null; type = types.nullOr types.optionSet;
        options = {
          order = mkOption (optional order);
          build = customHookType;
          push = hookType;
          activate = hookType;
          custom = mkOption { type = types.attrsOf types.optionSet; options = customHookType; };
        };
      };
    };


    # fileSystems = mkOption {
    #   options = { config, ... }: {
    #     options = {
    #       gce = mkOption {
    #         default = null;
    #         type = types.uniq (types.nullOr types.optionSet);
    #         options = gceDiskOptions;
    #         description = ''
    #           GCE disk to be attached to this mount point.  This is
    #           shorthand for defining a separate
    #           <option>deployment.gce.blockDeviceMapping</option>
    #           attribute.
    #         '';
    #       };
    #     };
    #     config = mkIf(config.gce != null) {
    #       device = mkDefault "${
    #           if config.gce.encrypt then "/dev/mapper/" else gce_dev_prefix
    #         }${
    #           get_disk_name (mkDefaultDiskName config.mountPoint config.gce)
    #       }";
    #     };
    #   };
    # };

  };

  config = {
    # deployment.targetHost = mkDefault config.networking.hostName;
    # deployment.sshPort = mkDefault (head config.services.openssh.ports);
    # fileSystems."/" = mkDefault ({ label = "/dev/disk/by-label/nixos"; fsType = "auto";});
  };

}
