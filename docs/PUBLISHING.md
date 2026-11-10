# Publishing this project to GitHub

Pick one.

## 1. The script (easiest if you have the GitHub CLI)

    cd carsim
    ./publish_to_github.sh                # creates github.com/<you>/carsim, public
    ./publish_to_github.sh carsim private # or a private one

It makes the first commit and creates and pushes the repository. It needs `git` and the
GitHub CLI (https://cli.github.com), logged in once with `gh auth login`. Without the CLI it
asks for the URL of an empty repository you made at https://github.com/new and pushes there.

## 2. Plain git

Create an empty repository at https://github.com/new (no README, no licence), then:

    cd carsim
    git init -b main
    git add -A
    git commit -m "carsim: analysis-first race car simulator"
    git remote add origin https://github.com/<you>/carsim.git
    git push -u origin main

## 3. In the browser, no git

Create a repository at https://github.com/new, choose "uploading an existing file", and drag
the contents of the `carsim` folder in. Hidden folders (`.github`, which holds the automatic
test run) may not show in your file browser: on a Mac press Cmd+Shift+. to show them.

After the first push, the Actions tab runs the tests automatically (about 8 minutes).
