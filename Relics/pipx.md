# Install pipx

**pipx** is a tool for installing and running Python applications in isolated environments.  
It keeps your system clean and lets you run [`Byzantium`](../Byzantium) or [`Hydra`](../Hydra) directly.

On most systems, installing pipx will also install Python automatically if it’s not already present.

---

## macOS

```bash
brew install pipx && pipx ensurepath
```

---

## Ubuntu / Debian

```bash
sudo apt install pipx
```

---

## Fedora / Red Hat

```bash
sudo dnf install pipx
```

---

## Arch / Manjaro

```bash
sudo pacman -S python-pipx
```

---

## Notes

- If pipx is not found after install, restart your terminal and try again  
- Linux users can use their package manager instead of pip if preferred  
- This should take less than 30 seconds on most systems  
