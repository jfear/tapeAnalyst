install:
	pip install -r requirements.txt

test:
	nosetests tests

run:
	./tapeAnalyst/analyzeTapeStationPng --gel ./data/JH_2.png \
		--sample ./data/JH_sample.csv \
		--output ./data/JH_report.html 
