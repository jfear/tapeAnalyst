install:
	pip install -r requirements.txt --user
	pip install git+https://github.com/alimuldal/diptest.git --user

run:
	python ./bin/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--range 300 700 \
		--report_name ${HOME}/Desktop/JH_report.html \
		--table_name ${HOME}/Desktop/JH_table.tsv

debug:
	python -m ipdb ./bin/analyzeTapeStationPng \
		--gel ./data/JH_gel.png \
		--sample ./data/JH_sample.csv \
		--range 300 700 \
		--report_name ${HOME}/Desktop/JH_report.html \
		--table_name ${HOME}/Desktop/JH_table.tsv
