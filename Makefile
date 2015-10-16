install:
	pip install -r requirements.txt

test:
	nosetests tests

run:
	./tapeAnalyst/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--output ./data/JH_report.html 
