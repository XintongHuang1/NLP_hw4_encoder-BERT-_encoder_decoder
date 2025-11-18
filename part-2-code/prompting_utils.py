import os
import re
import json


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    if not os.path.exists(schema_path):
        return None
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    # The model might generate responses in various formats, try multiple patterns
    
    # Pattern 1: SQL between ```sql and ```
    sql_pattern1 = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_pattern1, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: SQL between ``` and ```
    sql_pattern2 = r'```\s*(SELECT.*?)\s*```'
    match = re.search(sql_pattern2, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: SQL after "SQL:" or "Query:"
    sql_pattern3 = r'(?:SQL|Query):\s*(SELECT.*?)(?:\n|$)'
    match = re.search(sql_pattern3, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: Look for SELECT statement directly
    sql_pattern4 = r'(SELECT\s+.*?)(?:\n\n|\Z)'
    match = re.search(sql_pattern4, response, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        # Remove any trailing text that's not part of SQL
        # Stop at common delimiters
        for delimiter in ['\n\n', 'Note:', 'Example:', '```']:
            if delimiter in query:
                query = query.split(delimiter)[0].strip()
        return query
    
    # Pattern 5: If nothing else works, look for anything that starts with SELECT
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.upper().startswith('SELECT'):
            return line
    
    # Fallback: return the whole response if it contains SELECT
    if 'SELECT' in response.upper():
        return response.strip()
    
    # If no SQL found, return empty string
    return ""

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")