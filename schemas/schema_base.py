"""
Base classes for schema configuration system.
This allows the framework to support arbitrary document structures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable
import yaml
import os


class FieldConfig:
    """Configuration for a single field in the schema."""
    
    def __init__(self, name: str, field_type: str, evaluation: str, 
                 weight: float = 1.0, description: str = ""):
        self.name = name
        self.field_type = field_type  # 'categorical_dict', 'entity_list', 'text_dict', etc.
        self.evaluation = evaluation  # 'accuracy', 'f1', 'pairing', etc.
        self.weight = weight
        self.description = description
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FieldConfig':
        """Create FieldConfig from dictionary."""
        return cls(
            name=config_dict['name'],
            field_type=config_dict['type'],
            evaluation=config_dict['evaluation'],
            weight=config_dict.get('weight', 1.0),
            description=config_dict.get('description', '')
        )


class DocumentSchema(ABC):
    """Abstract base class for document schemas."""
    
    def __init__(self, schema_path: str = None, config_dict: Dict[str, Any] = None):
        """
        Initialize schema from YAML file or dictionary.
        
        Args:
            schema_path: Path to YAML schema configuration file
            config_dict: Dictionary containing schema configuration
        """
        if schema_path:
            with open(schema_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_dict:
            config = config_dict
        else:
            raise ValueError("Either schema_path or config_dict must be provided")
        
        self.schema_name = config.get('schema_name', 'unknown')
        self.version = config.get('version', 'v2')
        self.description = config.get('description', '')
        self.fields = [FieldConfig.from_dict(f) for f in config.get('fields', [])]
        self.prompt_template = config.get('prompt_template', '')
        
        # Build weights dictionary
        self.weights = {f.name: f.weight for f in self.fields}
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def get_field(self, field_name: str) -> FieldConfig:
        """Get field configuration by name."""
        for field in self.fields:
            if field.name == field_name:
                return field
        raise KeyError(f"Field '{field_name}' not found in schema")
    
    def get_evaluation_method(self, field_name: str) -> str:
        """Get evaluation method for a field."""
        return self.get_field(field_name).evaluation
    
    def get_weight(self, field_name: str) -> float:
        """Get weight for a field."""
        return self.weights.get(field_name, 0.0)
    
    def get_prompt(self) -> str:
        """Get the prompt template for this schema."""
        return self.prompt_template
    
    @abstractmethod
    def validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Validate that a prediction matches the expected schema.
        
        Args:
            prediction: Predicted data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def __repr__(self):
        return f"DocumentSchema(name='{self.schema_name}', version='{self.version}', fields={len(self.fields)})"


class MedicalFormSchema(DocumentSchema):
    """Schema for medical form documents."""
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Validate medical form prediction structure."""
        required_fields = {f.name for f in self.fields}
        pred_fields = set(prediction.keys())
        
        # Check if all required fields are present
        return required_fields.issubset(pred_fields)


class InvoiceSchema(DocumentSchema):
    """Schema for invoice documents."""
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Validate invoice prediction structure."""
        required_fields = {f.name for f in self.fields}
        pred_fields = set(prediction.keys())
        
        return required_fields.issubset(pred_fields)


class SchemaLoader:
    """Utility class to load schemas from configuration files."""
    
    SCHEMA_TYPES = {
        'medical_form': MedicalFormSchema,
        'invoice': InvoiceSchema,
        # Add more schema types as needed
    }
    
    @staticmethod
    def load_schema(schema_path: str) -> DocumentSchema:
        """
        Load a schema from a YAML file.
        
        Args:
            schema_path: Path to schema YAML file
            
        Returns:
            DocumentSchema instance
        """
        with open(schema_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        schema_type = config.get('schema_type', 'medical_form')
        schema_class = SchemaLoader.SCHEMA_TYPES.get(schema_type, DocumentSchema)
        
        return schema_class(schema_path=schema_path)
    
    @staticmethod
    def list_available_schemas(schemas_dir: str = None) -> List[str]:
        """List all available schema files in the schemas directory."""
        if schemas_dir is None:
            schemas_dir = os.path.dirname(__file__)
        
        schema_files = []
        for file in os.listdir(schemas_dir):
            if file.endswith('.yaml') and not file.startswith('_'):
                schema_files.append(file)
        
        return schema_files
