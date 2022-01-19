# Releases 

## How to do a release 

If you need to do a release, ensure all changes are reflected in `dev` branch. 
Also make sure that the changes have been manually tested!

If they have been manually tested, proceed to do the following: 


### Step 1: Updating the version

- In `relevanceai.__init__.py` update the version from `0.28.1` to `0.28.2` or whatever version you need it to be.

### Step 2: Updating Changelog

- On the Github repository home page (https://github.com/RelevanceAI/RelevanceAI), click on "releases"
- Create a new tag for the new version about to be released
- Click on Autogenerate changelogs
- Copy and paste the relevant changelogs into `docsrc/source/changelog.rst`
- Check out a new branch from development, commit the changes and then push
- Submit a PR and merge into Development

### Step 3: Creating Documentation

- Create a new branch from development for the version you are released following SemVer2.0 formatting, for example: `v0.28.0` or `v0.53.2`, this will automatically trigger `readthedocs` to create documentation once the new branch has been pushed.
- Submit a Pull Request for the new version branch (`v0.28.0`) into `main`
- Wait for tests to finish 

### Step 4: Releasing `Relevance AI` package
- Merge this PR, this will trigger a Github action that will update teh package.
- You will also want to do a Release on Github for fun.


TODO: Document how to release Conda packages (currently in the works).
