
import sqlite3
from pydantic import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")


def list_tables():
	c = conn.cursor()
	c.execute("SELECT name from sqlite_master WHERE type='table';")
	rows = c.fetchall()
	return "\n".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
	c = conn.cursor()
	try:
		c.execute(query)
		return c.fetchall()
	except sqlite3.OperationalError as err:
		return f"The following error occured: {str(err)}"


def describe_table(table_names):
	c = conn.cursor()
	tables = ', '.join("'" + table + "'" for table in table_names)
	rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type = 'table' and name in ({tables});")
	return "\n".join(row[0] for row in rows if row[0] is not None)


class RunQueryArgsSchema(BaseModel):
	query: str


run_query_tool = Tool.from_function(
	name="run_sqlite_query",
	description="run a sqlite query.",
	func = run_sqlite_query,
	args_schema=RunQueryArgsSchema
)

class DescribeTablesArgsSchema(BaseModel):
	tables_names: List[str]

desctibe_tables_tool = Tool.from_function(
	name="describe_tables",
	description="Given a list of tables, returns the schema of the tables",
	func=describe_table,
	args_schema=DescribeTablesArgsSchema

)