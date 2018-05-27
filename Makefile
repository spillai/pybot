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

python-develop:
	python setup.py develop

python-build:
	python setup.py sdist
	python setup.py bdist_wheel

python-install:
	python setup.py install

python-wheel:
	python setup.py bdist_wheel

python-dev-build:
	python setup.py build_ext --inplace

conda-build:
	conda build tools/conda.recipe

conda-install-runtime:
	conda create -q -n pybot-runtime-env -y
	conda install -c s_pillai pybot -n pybot-runtime-env -y

docker-build:
	docker build tools/docker

docker-push:
	docker tag pybot:latest
	docker push pybot:latest
