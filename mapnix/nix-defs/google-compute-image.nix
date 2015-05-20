# from: https://github.com/NixOS/nixpkgs/blob/master/nixos/modules/virtualisation/google-compute-image.nix

{diskSize, overrides}:
{ config, lib, pkgs, ... }:

with lib;
{
  imports = [
    <nixpkgs/nixos/modules/profiles/headless.nix>
    <nixpkgs/nixos/modules/profiles/qemu-guest.nix>
    { key = "userConfig"; config = overrides; }
  ];

  system.build.googleComputeImage =
    pkgs.vmTools.runInLinuxVM (
      pkgs.runCommand "google-compute-image"
        { preVM =
            ''
              mkdir $out
              diskImage=$out/$diskImageBase.raw
              truncate $diskImage --size ${diskSize}
              mv closure xchg/
            '';

          postVM =
            ''
              PATH=$PATH:${pkgs.gnutar}/bin:${pkgs.gzip}/bin
              pushd $out
              mv $diskImageBase.raw disk.raw
              tar -Szcf $diskImageBase.tar.gz disk.raw
              rm $out/disk.raw
              popd
            '';
          diskImageBase = "nixos";
          buildInputs = [ pkgs.utillinux pkgs.perl ];
          exportReferencesGraph =
            [ "closure" config.system.build.toplevel ];
        }
        ''
          # Create partition table
          ${pkgs.parted}/sbin/parted /dev/vda mklabel msdos
          ${pkgs.parted}/sbin/parted /dev/vda mkpart primary ext4 1 ${diskSize}
          ${pkgs.parted}/sbin/parted /dev/vda print
          . /sys/class/block/vda1/uevent
          mknod /dev/vda1 b $MAJOR $MINOR

          # Create an empty filesystem and mount it.
          ${pkgs.e2fsprogs}/sbin/mkfs.ext4 -L nixos /dev/vda1
          ${pkgs.e2fsprogs}/sbin/tune2fs -c 0 -i 0 /dev/vda1

          mkdir /mnt
          mount /dev/vda1 /mnt

          # The initrd expects these directories to exist.
          mkdir /mnt/dev /mnt/proc /mnt/sys

          mount --bind /proc /mnt/proc
          mount --bind /dev /mnt/dev
          mount --bind /sys /mnt/sys

          # Copy all paths in the closure to the filesystem.
          storePaths=$(perl ${pkgs.pathsFromGraph} /tmp/xchg/closure)

          mkdir -p /mnt/nix/store
          echo "copying everything (will take a while)..."
          cp -prd $storePaths /mnt/nix/store/

          # make sure the store is owned by root (in case we're being built
          # from a user-owned store)

          # XXX hacky - pretend there's a root user just so we can use `chown`.
          # Also, this will break stuff if there already was an /etc/passwd|group
          echo 'root:x:0:0:root:/root:/bin/bash' > /etc/passwd
          echo 'root:x:0:' > /etc/group
          chown -R root.root /mnt/nix/store

          # Register the paths in the Nix database.
          printRegistration=1 perl ${pkgs.pathsFromGraph} /tmp/xchg/closure | \
              chroot /mnt ${config.nix.package}/bin/nix-store --load-db

          # Create the system profile to allow nixos-rebuild to work.
          chroot /mnt ${config.nix.package}/bin/nix-env \
              -p /nix/var/nix/profiles/system --set ${config.system.build.toplevel}

          # `nixos-rebuild' requires an /etc/NIXOS.
          mkdir -p /mnt/etc
          touch /mnt/etc/NIXOS

          # `switch-to-configuration' requires a /bin/sh
          mkdir -p /mnt/bin
          ln -s ${config.system.build.binsh}/bin/sh /mnt/bin/sh

          # Generate the GRUB menu.
          ln -s vda /dev/sda
          chroot /mnt ${config.system.build.toplevel}/bin/switch-to-configuration boot

          umount /mnt/proc /mnt/dev /mnt/sys
          umount /mnt
        ''
    );

  fileSystems."/".label = "nixos";

  boot.kernelParams = [ "console=ttyS0" "panic=1" "boot.panic_on_fail" ];
  boot.initrd.kernelModules = [ "virtio_scsi" ];

  # Generate a GRUB menu.  Amazon's pv-grub uses this to boot our kernel/initrd.
  boot.loader.grub.device = "/dev/sda";
  boot.loader.grub.timeout = 0;

  # Don't put old configurations in the GRUB menu.  The user has no
  # way to select them anyway.
  boot.loader.grub.configurationLimit = 0;

  # Allow root logins only using the SSH key that the user specified
  # at instance creation time.
  services.openssh.enable = true;
  services.openssh.permitRootLogin = "without-password";

  # Force getting the hostname from Google Compute.
  networking.hostName = mkDefault "";

  # nothing but SSH runs on a bootstrap instance, so firewall just gets in the way
  networking.firewall.enable = false;

  # Configure default metadata hostnames
  networking.extraHosts = ''
    169.254.169.254 metadata.google.internal metadata
  '';

  environment.etc = [
    { source = <nixpkgs>; target="nixos/nixpkgs"; }
  ];

  networking.usePredictableInterfaceNames = false;
}
