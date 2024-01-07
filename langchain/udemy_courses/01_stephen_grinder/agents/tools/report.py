from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel

def write_report(filename, html):
	with open(filename, 'w') as f:
		f.write(html)

class WriteReportArgSchema(BaseModel):
	filename: str
	html: str


# difference between Tool and StructuredTool simply that Toll can 
# receive only one argument whereas StructuredTool can receive many

write_report_tool = StructuredTool.from_function(
	name="write_report",
	description="Write an HTML file to disk. Use this tool to write a report.",
	func=write_report,
	args_schema=WriteReportArgSchema
)