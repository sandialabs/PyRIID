coverage run -m unittest discover -s tests/ -p *.py -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed!" }
coverage report -i
coverage xml -i
