install:
	pip install -r requirements.txt

test:
	nosetests tests

run:
	python ./bin/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--range 300 700 \
		--output ./data/JH_report.html 

debug:
	python -m ipdb ./bin/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--range 300 700 \
		--output ./data/JH_report.html 
