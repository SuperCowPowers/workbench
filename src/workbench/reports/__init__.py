"""PDF reports composed from sections: model performance, data quality, data comparison."""

from workbench.reports import sections
from workbench.reports.paginator import Paginator, Panel
from workbench.reports.report import Note, Report, Section, Summary

__all__ = ["Note", "Paginator", "Panel", "Report", "Section", "Summary", "sections"]
