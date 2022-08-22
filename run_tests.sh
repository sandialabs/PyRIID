set -e
coverage run -m unittest tests/*.py -v
coverage report -i
coverage xml -i
