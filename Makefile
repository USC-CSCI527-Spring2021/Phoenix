MAKE_FILE_PATH=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR=$(dir $(MAKE_FILE_PATH))

check: generate_system_tests format lint tests

format:
	isort project/*
	black project/*

lint:
	isort --check-only project/*
	black --check project/*
	flake8 project/*

generate_system_tests:
	python project/system.py

tests:
	PYTHONPATH=./project pytest -n 4

tests_coverage:
	PYTHONPATH=./project pytest --cov=. --cov-report html -n 4

build_docker:
	docker build -t mahjong_bot .

GAMES=1
run_battle:
	docker run -u `id -u` -it --rm \
		--cpus=".9" \
		-v "$(CURRENT_DIR):/app/" \
		-v /dev/urandom:/dev/urandom \
		mahjong_bot python3 tenhou_env/project/bots_battle.py -g $(GAMES) $(ARGS)

run_stat:
	docker run -u `id -u` -it --rm \
        --cpus=".9" \
        --memory="4g" \
		-v "$(CURRENT_DIR)tenhou_env/project/:/app/" \
		-v "$(db_folder):/app/statistics/db/" \
		mahjong_bot pypy3 tenhou_env/run_stat.py -p /app/statistics/db/$(file_name)

run_on_tenhou:
	docker-compose -f tenhou_env/docker-compose.yaml up

archive_replays:
	tar -czvf "logs-$(shell date '+%Y-%m-%d-%H-%M').tar.gz" -C ./project/battle_results/logs/ .
	tar -czvf "replays-$(shell date '+%Y-%m-%d-%H-%M').tar.gz" -C ./project/battle_results/replays/ .
