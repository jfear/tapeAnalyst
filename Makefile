install:
	pip install -r requirements.txt
	pip install git+https://github.com/alimuldal/diptest.git

run:
	python ./bin/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--range 300 700 \
		--output ${HOME}/Desktop/JH_report.html 

debug:
	python -m ipdb ./bin/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--range 300 700 \
		--output ${HOME}/Desktop/JH_report.html 
