# Faris!-SpotMe

SUTD term 1-2 IDEA project

## Team members

Faris
Sal
Jia Wei
Joel
Adam

## Hardware requirements

- raspberry pi 5 4GB RAM
- power supply for raspberry pi 5 (output 5V 5A)
- touchscreen display monitor
- micro hdmi cable to display to connect the rpi to the display
- usb c a cable for connection between rpi and touchscreen to relay touch data
- usb camera

## Raspberry PI set up

follow the documentation @ https://www.raspberrypi.com/documentation/computers/getting-started.html

## Ultralytics YOLO set up

follow the documentation @ https://docs.ultralytics.com/guides/raspberry-pi/

for this project, we will be using yolo26n which is optimised for edge devices such as the raspberry pi

further optimise the model to an ncnn model to optimise it for use on raspberry pi

if doing development on your own pc/laptop, just use the normal `.pt` pytorch model.

## Developer Guide

### Clone the repo

```
git clone https://github.com/StrwBrriShrtcke/Faris-Spotme.git
```

```
cd Faris-SpotMe
```

### First time set up

```
# Set your name & email (shows up on commits)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# (Optional) make 'main' the default branch name on new repos
git config --global init.defaultBranch main
```

### Pull the latest code (save your work before doing this)

```
git pull
```

### Prep your files to commit

```
# Add everything you changed
git add .

# or add specific files
git add src/app.py README.md
```

### Commit your changes with a message

```
git commit -m "Brief description of what you changed"
```

### Push your changes to Github

```
git push

# if this is your first push on a new branch, you might see: fatal: The current branch xyz has no upstream branch.
# run the below once then subsequently just run git push
git push --set-upstream origin main
```

### if conflict on git pull

```
# run git stash to stash your changes
git stash

# run git pull to pull latest changes to codebase
git pull

# run git stash pop to unstash your changes
git stash pop

# after that handle merge conflicts and proceed to re-add, re-commit then push your changes
```

### Download UV

follow the documentation @ https://docs.astral.sh/uv/getting-started/installation/

### First run

make sure you have the latest code by running `git pull`

update your dependencies by running, do this step before running the app for the first time and if your code happens to not run, then update your dependencies again by running the below command

```
uv sync
```

you can now run the app with

```
uv run main/app.py
```

go to http://localhost:3000 to view the app
