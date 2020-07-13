prepare-dev:
	pip install --upgrade pip && \
	pip install -U -r requirements.txt && \
	pip setup.py develop

test:
	pytest -v tests/*_test.py

clean:
	rm -rf build/; rm -rf dist/; rm -rf *.egg-info

build:
	python setup.py sdist

upload:
	twine upload dist/*

build-upload:
	make build && make upload