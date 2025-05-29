"""Data models for Tree-sitter analysis output."""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

class Import(BaseModel):
    """Model for import statements."""
    name: str
    source: str
    items: Optional[List[str]] = None

class Parameter(BaseModel):
    """Model for function/method parameters."""
    name: str
    type: Optional[str] = None

class BaseClass(BaseModel):
    """Model for base classes."""
    name: str
    access: Optional[Literal["public", "protected", "private"]] = None

class Method(BaseModel):
    """Model for class methods."""
    name: str
    parameters: Optional[List[Parameter]] = None
    calls: Optional[List[str]] = None
    docstring: Optional[str] = None

class Attribute(BaseModel):
    """Model for class attributes."""
    name: str
    type: Optional[str] = None
    static: bool = False
    docstring: Optional[str] = None

class Class(BaseModel):
    """Model for class definitions."""
    name: str
    bases: Optional[List[BaseClass]] = None
    methods: Optional[List[Method]] = None
    attributes: Optional[List[Attribute]] = None
    docstring: Optional[str] = None

class Function(BaseModel):
    """Model for top-level functions."""
    name: str
    parameters: Optional[List[Parameter]] = None
    calls: Optional[List[str]] = None
    docstring: Optional[str] = None

class Variable(BaseModel):
    """Model for top-level variables."""
    name: str
    kind: Literal["constant", "type_alias", "external_variable"]
    type: Optional[str] = None
    value: Optional[str] = None
    docstring: Optional[str] = None

class EnumMember(BaseModel):
    """Model for enum members."""
    name: str
    value: Optional[str] = None
    docstring: Optional[str] = None

class Enum(BaseModel):
    """Model for enum definitions."""
    name: str
    members: Optional[List[EnumMember]] = None
    docstring: Optional[str] = None

class FileAnalysis(BaseModel):
    """Model for complete file analysis."""
    file_path: str
    analyzer: str
    imports: Optional[List[Import]] = None
    classes: Optional[List[Class]] = None
    functions: Optional[List[Function]] = None
    variables: Optional[List[Variable]] = None
    enums: Optional[List[Enum]] = None

class FailedAnalysis(BaseModel):
    """Model for failed file analysis."""
    file_path: str
    analyzer: str
    reason: str