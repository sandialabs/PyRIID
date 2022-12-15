set -e
coverage run --source=./riid -m unittest tests/*.py
coverage report -i --skip-empty
