.PHONY: install run dev lint clean

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8081

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload