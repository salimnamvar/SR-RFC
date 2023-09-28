""" Python to PostgreSQL Mapper

"""


def map_type(a_pytype):
    # Define a mapping from Python/NumPy data types to PostgreSQL data types
    data_type_mapping = {
        'int64': 'bigint',
        'int32': 'integer',
        'int16': 'smallint',
        'float64': 'double precision',
        'float32': 'real',
        'object': 'text',
        'string': 'text',
        'bool': 'boolean',
        'datetime64[ns]': 'timestamp without time zone',
        'datetime64[ns, UTC]': 'timestamp with time zone',
        'timedelta64[ns]': 'interval',
        'bytes': 'bytea',
        # Add more mappings for other data types
    }

    # Check if the data type is in the mapping, and return the corresponding PostgreSQL data type
    return data_type_mapping.get(a_pytype, 'text')