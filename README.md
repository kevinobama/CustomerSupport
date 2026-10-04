installation:
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Now install the packages
pip install fastapi uvicorn

uvicorn main:app --host 0.0.0.0 --port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

