import sqlite3
from echo.utils import (
    get_db_name, 
    serialize_dict, 
    deserialize_dict
)



def create_db() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_name())
    return conn


def create_table(query: str) -> None:
    conn = create_db()
    conn.execute(query)
    conn.commit()
    conn.close()


def insert_record(
    table_name: str,
    attributes: dict,
) -> None:
    """
    Checks if the record already exists in the database.
    If it does, then no new record is inserted.
    """
    condition_attributes = {k: v for k, v in attributes.items() if not isinstance(v, (list, dict))}
    attributes = serialize_dict(attributes)
    conn = create_db()
    
    # Check if the record already exists
    query = f"SELECT * FROM {table_name} WHERE "\
        + " AND ".join(f"{key} = ?" for key in condition_attributes.keys())
    
    record = conn.execute(query, tuple(condition_attributes.values())).fetchone()
    if record:
        conn.close()
        return
    
    query = f"INSERT INTO {table_name} ("\
        + ", ".join(attributes.keys())\
        + ") VALUES ("\
        + ", ".join("?" for _ in attributes)\
        + ")"
        
    conn.execute(query, tuple(attributes.values()))
    conn.commit()
    conn.close()


def update_record(table_name: str, condition_dict: dict, update_dict: dict) -> None:
    condition_dict = serialize_dict(condition_dict)
    update_dict = serialize_dict(update_dict)
    
    conn = create_db()
    if not update_dict:
        return
    query = f"UPDATE {table_name} SET "\
        + ", ".join(f"{key} = ?" for key in update_dict.keys())\
        + " WHERE " + " AND ".join(f"{key} = ?" for key in condition_dict.keys())
        
    conn.execute(query, tuple(update_dict.values()) + tuple(condition_dict.values()))
    conn.commit()
    conn.close()
    

def delete_record(table_name, conditions_dict: dict) -> None:
    conditions_dict = serialize_dict(conditions_dict)
    conn = create_db()
    query = f"DELETE FROM {table_name} WHERE "\
        + " AND ".join(f"{key} = ?" for key in conditions_dict.keys())
        
    conn.execute(query, tuple(conditions_dict.values()))
    conn.commit()
    conn.close()
    


def query_records(table_name, condition_dict: dict = None, limit: int = None):
    if condition_dict is None:
        condition_dict = {1: 1}
        
    condition_dict = serialize_dict(condition_dict)
    conn = create_db()
    query = f"SELECT * FROM {table_name} WHERE "\
        + " AND ".join(f"{key} = ?" for key in condition_dict.keys())\
        + " ORDER BY timestamp DESC"
        
    if limit:
        query += f" LIMIT {limit}"
    cursor = conn.execute(query, tuple(condition_dict.values()))
    col_names = [desc[0] for desc in cursor.description]
    records = cursor.fetchall()
    conn.close()
    records = [deserialize_dict(dict(zip(col_names, record))) for record in records]
    return records


def check_record_exists(table_name, condition_dict: dict):
    conn = create_db()
    query = f"SELECT * FROM {table_name} WHERE "\
        + " AND ".join(f"{key} = ?" for key in condition_dict.keys())
        
    record = conn.execute(query, tuple(condition_dict.values())).fetchone()
    conn.close()
    return record is not None


def get_record(table_name, condition_dict: dict):
    conn = create_db()
    query = f"SELECT * FROM {table_name} WHERE "\
        + " AND ".join(f"{key} = ?" for key in condition_dict.keys())
    
    cursor = conn.execute(query, tuple(condition_dict.values()))
    record = cursor.fetchone()
    if not record:
        return None
    
    col_names = [desc[0] for desc in cursor.description]
    record_dict = dict(zip(col_names, record))
    conn.close()
    return deserialize_dict(record_dict)
    

def get_table_columns(table):
    conn = create_db()
    query = f"PRAGMA table_info({table})"
    columns = conn.execute(query).fetchall()
    ### Exclude id and timestamp columns
    columns = [col[1] for col in columns if col[1] not in ["id", "timestamp"]]
    conn.close()
    return columns    