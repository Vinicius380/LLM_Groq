from sqlalchemy import create_engine

MYSQL_USER = 'root'
MYSQL_PASSWORD = '525748'
MYSQL_HOST = 'localhost'
MYSQL_PORT = '3306'
MYSQL_DB_NAME = 'bd_vibracao_motoreletrico'

DATABASE_URL = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB_NAME}'
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print(result.fetchone())