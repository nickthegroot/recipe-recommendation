#!/usr/bin/fish

# Install exa (https://github.com/ogham/exa)
apt update
apt install -y exa htop

# Install fisher
fisher install IlanCosman/tide@v5
fisher install gazorby/fish-exa