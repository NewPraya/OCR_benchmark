"""
Schema configuration system for document structure definition.
"""

from .schema_base import (
    FieldConfig,
    DocumentSchema,
    MedicalFormSchema,
    InvoiceSchema,
    SchemaLoader
)

__all__ = [
    'FieldConfig',
    'DocumentSchema',
    'MedicalFormSchema',
    'InvoiceSchema',
    'SchemaLoader'
]
