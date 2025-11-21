"""
Static Code Analysis for CPU Training System

Analyzes the code structure, identifies innovations, and validates
the implementation without requiring PyTorch to be installed.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple


class CodeAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.source = Path(file_path).read_text()
        try:
            self.tree = ast.parse(self.source)
        except SyntaxError as e:
            self.tree = None
            self.syntax_error = str(e)

    def analyze_structure(self) -> Dict:
        """Analyze the overall code structure"""
        if not self.tree:
            return {"error": self.syntax_error}

        results = {
            "classes": [],
            "functions": [],
            "imports": [],
            "line_count": len(self.source.split('\n')),
            "docstrings": []
        }

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                results["classes"].append({
                    "name": node.name,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "bases": [self._get_node_name(b) for b in node.bases],
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node)
                })
                if ast.get_docstring(node):
                    results["docstrings"].append((node.name, ast.get_docstring(node)))

            elif isinstance(node, ast.FunctionDef):
                if not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    results["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    })
                    if ast.get_docstring(node):
                        results["docstrings"].append((node.name, ast.get_docstring(node)))

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    results["imports"].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    results["imports"].append(f"{module}.{alias.name}")

        return results

    def _get_node_name(self, node):
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        return str(node)

    def identify_innovations(self) -> Dict:
        """Identify the key algorithmic innovations"""
        innovations = {
            "fused_layers": False,
            "bf16_mixed_precision": False,
            "lamb_optimizer": False,
            "lookahead_optimizer": False,
            "dynamic_freezing": False,
            "gradient_accumulation": False,
            "synthetic_dataset": False
        }

        # Check for FusedLinearReLU
        if "FusedLinearReLU" in self.source:
            innovations["fused_layers"] = True

        # Check for BF16 usage
        if "bfloat16" in self.source or "torch.bfloat16" in self.source:
            innovations["bf16_mixed_precision"] = True

        # Check for LAMB optimizer
        if "LAMB" in self.source or "class LAMB_BF16" in self.source:
            innovations["lamb_optimizer"] = True

        # Check for Lookahead
        if "Lookahead" in self.source or "class Lookahead" in self.source:
            innovations["lookahead_optimizer"] = True

        # Check for dynamic freezing
        if "DynamicFreezer" in self.source or "frozen" in self.source:
            innovations["dynamic_freezing"] = True

        # Check for gradient accumulation
        if "accumulation" in self.source.lower():
            innovations["gradient_accumulation"] = True

        # Check for synthetic dataset
        if "SyntheticDataset" in self.source:
            innovations["synthetic_dataset"] = True

        return innovations

    def check_potential_issues(self) -> List[Dict]:
        """Identify potential issues in the code"""
        issues = []

        # Check for hardcoded values
        hardcoded_patterns = [
            (r'torch\.set_num_threads\((\d+)\)', "Hardcoded thread count"),
            (r'lr=([0-9.]+)', "Hardcoded learning rate"),
            (r'batch_size=(\d+)', "Hardcoded batch size"),
        ]

        for pattern, description in hardcoded_patterns:
            matches = re.findall(pattern, self.source)
            if matches:
                issues.append({
                    "type": "hardcoded_value",
                    "description": description,
                    "values": matches
                })

        # Check for BF16 support check
        if "torch.cpu.amp.autocast_supported()" not in self.source:
            issues.append({
                "type": "missing_check",
                "description": "Code uses BF16 but doesn't verify CPU support",
                "severity": "warning"
            })

        # Check for error handling
        if "try:" not in self.source or "except" not in self.source:
            issues.append({
                "type": "missing_error_handling",
                "description": "No exception handling found",
                "severity": "warning"
            })

        # Check for parameter validation
        if "assert" not in self.source and "raise" not in self.source:
            issues.append({
                "type": "missing_validation",
                "description": "No input validation found",
                "severity": "info"
            })

        return issues

    def analyze_complexity(self) -> Dict:
        """Analyze code complexity metrics"""
        if not self.tree:
            return {}

        complexity = {
            "total_classes": 0,
            "total_functions": 0,
            "total_methods": 0,
            "max_method_length": 0,
            "avg_method_length": 0,
        }

        method_lengths = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                complexity["total_classes"] += 1
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        complexity["total_methods"] += 1
                        length = self._count_lines(item)
                        method_lengths.append(length)
                        complexity["max_method_length"] = max(
                            complexity["max_method_length"], length
                        )

            elif isinstance(node, ast.FunctionDef):
                if not hasattr(node, 'parent'):
                    complexity["total_functions"] += 1

        if method_lengths:
            complexity["avg_method_length"] = sum(method_lengths) / len(method_lengths)

        return complexity

    def _count_lines(self, node):
        """Count lines in an AST node"""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return node.end_lineno - node.lineno + 1
        return 0

    def validate_innovations(self) -> Dict:
        """Validate that claimed innovations are properly implemented"""
        validations = {}

        # Validate Fused Layer implementation
        if "FusedLinearReLU" in self.source:
            has_init = "__init__" in self.source and "self.weight" in self.source
            has_forward = "def forward" in self.source and "torch.relu_" in self.source
            uses_bf16 = "torch.bfloat16" in self.source
            validations["fused_layer"] = {
                "implemented": True,
                "has_proper_init": has_init,
                "has_forward_method": has_forward,
                "uses_bf16": uses_bf16,
                "valid": has_init and has_forward and uses_bf16
            }

        # Validate LAMB optimizer
        if "LAMB_BF16" in self.source:
            has_step = "def step" in self.source
            has_trust_ratio = "trust_ratio" in self.source
            uses_bf16_states = "torch.bfloat16" in self.source and "exp_avg" in self.source
            validations["lamb_optimizer"] = {
                "implemented": True,
                "has_step_method": has_step,
                "has_trust_ratio": has_trust_ratio,
                "uses_bf16_states": uses_bf16_states,
                "valid": has_step and has_trust_ratio and uses_bf16_states
            }

        # Validate Lookahead
        if "Lookahead" in self.source:
            has_slow_weights = "'slow'" in self.source
            has_interpolation = "state['slow'].add_" in self.source
            validations["lookahead"] = {
                "implemented": True,
                "has_slow_weights": has_slow_weights,
                "has_interpolation": has_interpolation,
                "valid": has_slow_weights and has_interpolation
            }

        # Validate Dynamic Freezing
        if "DynamicFreezer" in self.source:
            has_hooks = "register_hook" in self.source
            has_gradient_tracking = "grad.norm()" in self.source
            can_freeze = "requires_grad = False" in self.source
            validations["dynamic_freezing"] = {
                "implemented": True,
                "has_gradient_hooks": has_hooks,
                "tracks_gradients": has_gradient_tracking,
                "can_freeze_layers": can_freeze,
                "valid": has_hooks and has_gradient_tracking and can_freeze
            }

        return validations


def print_report(analyzer: CodeAnalyzer):
    """Generate and print comprehensive analysis report"""
    print("=" * 80)
    print("CPU TRAINING SYSTEM - STATIC CODE ANALYSIS REPORT")
    print("=" * 80)

    # Structure analysis
    print("\n1. CODE STRUCTURE")
    print("-" * 80)
    structure = analyzer.analyze_structure()

    if "error" in structure:
        print(f"ERROR: {structure['error']}")
        return

    print(f"Total lines: {structure['line_count']}")
    print(f"Total classes: {len(structure['classes'])}")
    print(f"Total functions: {len(structure['functions'])}")
    print(f"Total imports: {len(structure['imports'])}")
    print(f"Documented components: {len(structure['docstrings'])}")

    print("\nClasses defined:")
    for cls in structure['classes']:
        print(f"  - {cls['name']} (line {cls['line']})")
        print(f"    Methods: {', '.join(cls['methods'])}")
        if cls['docstring']:
            print(f"    Docstring: {cls['docstring'][:60]}...")

    # Innovation identification
    print("\n2. ALGORITHMIC INNOVATIONS")
    print("-" * 80)
    innovations = analyzer.identify_innovations()
    for name, present in innovations.items():
        status = "✓ FOUND" if present else "✗ MISSING"
        print(f"  {status}: {name.replace('_', ' ').title()}")

    # Innovation validation
    print("\n3. INNOVATION VALIDATION")
    print("-" * 80)
    validations = analyzer.validate_innovations()
    for name, details in validations.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        for key, value in details.items():
            if key != "valid":
                status = "✓" if value else "✗"
                print(f"    {status} {key.replace('_', ' ').title()}: {value}")
        overall = "VALID ✓" if details.get("valid", False) else "INVALID ✗"
        print(f"    Overall: {overall}")

    # Complexity analysis
    print("\n4. COMPLEXITY METRICS")
    print("-" * 80)
    complexity = analyzer.analyze_complexity()
    print(f"Total classes: {complexity.get('total_classes', 0)}")
    print(f"Total functions: {complexity.get('total_functions', 0)}")
    print(f"Total methods: {complexity.get('total_methods', 0)}")
    print(f"Max method length: {complexity.get('max_method_length', 0)} lines")
    print(f"Avg method length: {complexity.get('avg_method_length', 0):.1f} lines")

    # Potential issues
    print("\n5. POTENTIAL ISSUES")
    print("-" * 80)
    issues = analyzer.check_potential_issues()
    if not issues:
        print("No critical issues found.")
    else:
        for i, issue in enumerate(issues, 1):
            severity = issue.get('severity', 'info').upper()
            print(f"\n{i}. [{severity}] {issue['description']}")
            if 'values' in issue:
                print(f"   Found: {issue['values']}")

    # Summary
    print("\n6. SUMMARY")
    print("-" * 80)
    total_innovations = sum(innovations.values())
    total_valid = sum(1 for v in validations.values() if v.get("valid", False))

    print(f"Innovations found: {total_innovations}/{len(innovations)}")
    print(f"Innovations validated: {total_valid}/{len(validations)}")
    print(f"Issues found: {len(issues)}")

    if total_innovations == len(innovations) and total_valid == len(validations):
        print("\n✓ CODE APPEARS STRUCTURALLY SOUND")
    else:
        print("\n⚠ CODE NEEDS REVIEW")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyzer = CodeAnalyzer("cpu_training_system.py")
    print_report(analyzer)
