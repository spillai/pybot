all: clean build

clean:
	-$(RM) -rf dist
	-$(RM) -rf build
	-$(RM) -rf pybot.egg-info
	-$(RM) -rf cmake_build
	-$(RM) -f .env
	find . | grep '\.so' | xargs echo

clean-wheel:
	-$(RM) -rf dist
	-$(RM) -rf build
	-$(RM) -rf pybot.egg-info

develop:
	python setup.py develop

build:
	python setup.py sdist
	python setup.py bdist_wheel

install:
	python setup.py install

wheel:
	python setup.py bdist_wheel

conda-build:
	conda build tools/conda.recipe

conda-install-runtime:
	conda create -q -n pybot-runtime-env -y
	conda install -c s_pillai pybot -n pybot-runtime-env -y

anaconda-push:
	./scripts/deploy_anaconda.sh

dev-build:
	python setup.py build_ext --inplace

docker-build:
	docker build docker/

docker-push:
	docker tag pybot:latest
	docker push pybot:latest
