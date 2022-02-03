# mkdir channel
# mkdir channel/linux-64
# mkdir channel/linux-32
# mkdir channel/osx-64/
# mkdir channel/win-64/
# mkdir channel/win-32
# cp dist/* channel/linux-64/
# cp dist/* channel/linux-32/
# cp dist/* channel/osx-64/
# cp dist/* channel/win-64/
# cp dist/* channel/win-32/

# conda convert ~/miniconda/conda-bld/osx-64/cookiecutter-0.9.1_BUILDNUM.tar.bz2 -p all
conda convert /Users/jacky.wong/opt/anaconda3/conda-bld/osx-64/relevanceai-0.28.0-py39_0.tar.bz2 -p all channel

