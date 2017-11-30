clean:
	-$(RM) -rf dist
	-$(RM) -rf build
	-$(RM) -rf pybot.egg-info
	-$(RM) -rf cmake_build
	-$(RM) -f .env

build: 
	python setup.py sdist
	python setup.py bdist_wheel

wheel:
	python setup.py bdist_wheel	

conda-build:
	conda build tools/conda

all: clean build
