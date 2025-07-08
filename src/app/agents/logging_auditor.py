"""
Logging Auditor Agent

This agent maintains a detailed audit log of every step in the pipeline and enforces
compliance with any constraints (such as content or security policies).
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

from ..core.state import MASXState
from ..core.exceptions import AgentException, ValidationError
from ..services.database import DatabaseService
from .base import BaseAgent


class LoggingAuditor(BaseAgent):
    """
    Logging Auditor Agent for maintaining audit trails and compliance monitoring.

    This agent:
    - Tracks agent invocations, tool usage, outputs, and execution time
    - Enforces compliance with content and security policies
    - Maintains structured logs for traceability and debugging
    - Monitors for anomalies and policy violations
    """

    def __init__(self, database_service: Optional[DatabaseService] = None):
        """Initialize the Logging Auditor agent."""
        super().__init__("logging_auditor")
        self.database_service = database_service or DatabaseService()
        self.logger = logging.getLogger(__name__)

        # Define content policies
        self.content_policies = {
            "forbidden_keywords": [
                "classified",
                "secret",
                "confidential",
                "top secret",
                "restricted",
                "internal use only",
                "sensitive",
            ],
            "max_content_length": 10000,
            "required_fields": ["timestamp", "agent", "action", "status"],
        }

    def log_agent_execution(
        self,
        agent_name: str,
        action: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: float,
        status: str = "success",
    ) -> str:
        """
        Log an agent execution event.

        Args:
            agent_name: Name of the agent that executed
            action: Action performed by the agent
            parameters: Input parameters for the action
            result: Output result from the action
            execution_time: Time taken for execution in seconds
            status: Execution status (success/failure)

        Returns:
            ID of the logged entry
        """
        try:
            # Create log entry
            log_entry = {
                "agent": agent_name,
                "action": action,
                "parameters": self._sanitize_parameters(parameters),
                "result": self._sanitize_result(result),
                "execution_time": execution_time,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "log_hash": self._generate_log_hash(
                    agent_name, action, parameters, result
                ),
            }

            # Store in database
            log_id = self.database_service.insert_audit_log(log_entry)

            self.logger.info(
                f"Logged execution for {agent_name}.{action} (ID: {log_id})"
            )
            return log_id

        except Exception as e:
            self.logger.error(f"Failed to log agent execution: {str(e)}")
            raise AgentException(f"Agent execution logging failed: {str(e)}")

    def validate_content_policy(
        self, content: str, content_type: str = "text"
    ) -> Dict[str, Any]:
        """
        Validate content against defined policies.

        Args:
            content: Content to validate
            content_type: Type of content (text/json/url)

        Returns:
            Dict containing validation results
        """
        try:
            validation_results = {"valid": True, "violations": [], "warnings": []}

            # Check content length
            if len(content) > self.content_policies["max_content_length"]:
                validation_results["warnings"].append(
                    {
                        "type": "content_length",
                        "message": f"Content exceeds maximum length of {self.content_policies['max_content_length']} characters",
                    }
                )

            # Check for forbidden keywords
            content_lower = content.lower()
            found_keywords = []
            for keyword in self.content_policies["forbidden_keywords"]:
                if keyword in content_lower:
                    found_keywords.append(keyword)

            if found_keywords:
                validation_results["violations"].append(
                    {
                        "type": "forbidden_keywords",
                        "keywords": found_keywords,
                        "message": "Content contains forbidden keywords",
                    }
                )
                validation_results["valid"] = False

            # Check for potential sensitive data patterns
            sensitive_patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit card pattern
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
            ]

            import re

            for pattern in sensitive_patterns:
                if re.search(pattern, content):
                    validation_results["warnings"].append(
                        {
                            "type": "sensitive_data_pattern",
                            "pattern": pattern,
                            "message": "Content may contain sensitive data",
                        }
                    )

            return validation_results

        except Exception as e:
            self.logger.error(f"Error during content policy validation: {str(e)}")
            return {
                "valid": False,
                "violations": [{"type": "validation_error", "message": str(e)}],
                "warnings": [],
            }

    def validate_json_schema(
        self, data: Dict[str, Any], expected_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate JSON data against expected schema.

        Args:
            data: JSON data to validate
            expected_schema: Expected schema structure

        Returns:
            Dict containing validation results
        """
        try:
            validation_results = {
                "valid": True,
                "errors": [],
                "missing_fields": [],
                "extra_fields": [],
            }

            # Check required fields
            for field, field_type in expected_schema.items():
                if field not in data:
                    validation_results["missing_fields"].append(field)
                    validation_results["valid"] = False
                elif not isinstance(data[field], field_type):
                    validation_results["errors"].append(
                        {
                            "field": field,
                            "expected": field_type.__name__,
                            "actual": type(data[field]).__name__,
                        }
                    )
                    validation_results["valid"] = False

            # Check for extra fields
            for field in data:
                if field not in expected_schema:
                    validation_results["extra_fields"].append(field)

            return validation_results

        except Exception as e:
            self.logger.error(f"Error during JSON schema validation: {str(e)}")
            return {
                "valid": False,
                "errors": [{"type": "validation_error", "message": str(e)}],
                "missing_fields": [],
                "extra_fields": [],
            }

    def monitor_agent_outputs(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor agent outputs for anomalies and policy violations.

        Args:
            agent_outputs: Dictionary of agent outputs to monitor

        Returns:
            Dict containing monitoring results
        """
        try:
            monitoring_results = {
                "anomalies": [],
                "policy_violations": [],
                "quality_issues": [],
                "overall_status": "healthy",
            }

            for agent_name, output in agent_outputs.items():
                # Check for empty or null outputs
                if not output or output.get("status") == "failed":
                    monitoring_results["quality_issues"].append(
                        {
                            "agent": agent_name,
                            "issue": "empty_or_failed_output",
                            "severity": "high",
                        }
                    )

                # Check output content for policy violations
                if isinstance(output, dict):
                    output_str = json.dumps(output)
                    content_validation = self.validate_content_policy(output_str)

                    if not content_validation["valid"]:
                        monitoring_results["policy_violations"].append(
                            {
                                "agent": agent_name,
                                "violations": content_validation["violations"],
                                "severity": "medium",
                            }
                        )

                    if content_validation["warnings"]:
                        monitoring_results["anomalies"].append(
                            {
                                "agent": agent_name,
                                "warnings": content_validation["warnings"],
                                "severity": "low",
                            }
                        )

                # Check for execution time anomalies
                if isinstance(output, dict) and "execution_time" in output:
                    execution_time = output["execution_time"]
                    if execution_time > 300:  # 5 minutes threshold
                        monitoring_results["anomalies"].append(
                            {
                                "agent": agent_name,
                                "issue": "long_execution_time",
                                "execution_time": execution_time,
                                "severity": "medium",
                            }
                        )

            # Determine overall status
            if monitoring_results["policy_violations"]:
                monitoring_results["overall_status"] = "policy_violation"
            elif monitoring_results["quality_issues"]:
                monitoring_results["overall_status"] = "quality_issue"
            elif monitoring_results["anomalies"]:
                monitoring_results["overall_status"] = "anomaly_detected"

            return monitoring_results

        except Exception as e:
            self.logger.error(f"Error during agent output monitoring: {str(e)}")
            raise AgentException(f"Agent output monitoring failed: {str(e)}")

    def generate_audit_report(
        self, workflow_id: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report for a workflow execution.

        Args:
            workflow_id: ID of the workflow
            start_time: Start time of the workflow
            end_time: End time of the workflow

        Returns:
            Dict containing audit report
        """
        try:
            self.logger.info(f"Generating audit report for workflow: {workflow_id}")

            # Retrieve logs for the workflow
            logs = self.database_service.get_audit_logs_for_workflow(
                workflow_id, start_time.isoformat(), end_time.isoformat()
            )

            # Analyze logs
            total_executions = len(logs)
            successful_executions = sum(
                1 for log in logs if log.get("status") == "success"
            )
            failed_executions = total_executions - successful_executions

            # Calculate execution times
            execution_times = [log.get("execution_time", 0) for log in logs]
            avg_execution_time = (
                sum(execution_times) / len(execution_times) if execution_times else 0
            )
            max_execution_time = max(execution_times) if execution_times else 0

            # Group by agent
            agent_stats = {}
            for log in logs:
                agent = log.get("agent", "unknown")
                if agent not in agent_stats:
                    agent_stats[agent] = {
                        "executions": 0,
                        "successful": 0,
                        "failed": 0,
                        "total_time": 0,
                    }

                agent_stats[agent]["executions"] += 1
                if log.get("status") == "success":
                    agent_stats[agent]["successful"] += 1
                else:
                    agent_stats[agent]["failed"] += 1
                agent_stats[agent]["total_time"] += log.get("execution_time", 0)

            # Calculate success rates
            for agent in agent_stats:
                stats = agent_stats[agent]
                stats["success_rate"] = (
                    stats["successful"] / stats["executions"]
                    if stats["executions"] > 0
                    else 0
                )
                stats["avg_time"] = (
                    stats["total_time"] / stats["executions"]
                    if stats["executions"] > 0
                    else 0
                )

            audit_report = {
                "workflow_id": workflow_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": (
                    successful_executions / total_executions
                    if total_executions > 0
                    else 0
                ),
                "avg_execution_time": avg_execution_time,
                "max_execution_time": max_execution_time,
                "agent_statistics": agent_stats,
                "generated_at": datetime.utcnow().isoformat(),
            }

            self.logger.info("Audit report generated successfully")
            return audit_report

        except Exception as e:
            self.logger.error(f"Error generating audit report: {str(e)}")
            raise AgentException(f"Audit report generation failed: {str(e)}")

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging (remove sensitive data)."""
        try:
            sanitized = {}
            sensitive_keys = ["password", "token", "key", "secret", "api_key"]

            for key, value in parameters.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, str) and len(value) > 100:
                    sanitized[key] = value[:100] + "..."
                else:
                    sanitized[key] = value

            return sanitized

        except Exception as e:
            self.logger.warning(f"Failed to sanitize parameters: {str(e)}")
            return {"error": "sanitization_failed"}

    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize result for logging (remove sensitive data)."""
        try:
            sanitized = {}
            sensitive_keys = ["password", "token", "key", "secret", "api_key"]

            for key, value in result.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, str) and len(value) > 200:
                    sanitized[key] = value[:200] + "..."
                else:
                    sanitized[key] = value

            return sanitized

        except Exception as e:
            self.logger.warning(f"Failed to sanitize result: {str(e)}")
            return {"error": "sanitization_failed"}

    def _generate_log_hash(
        self,
        agent_name: str,
        action: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
    ) -> str:
        """Generate a hash for the log entry."""
        try:
            log_string = f"{agent_name}:{action}:{json.dumps(parameters, sort_keys=True)}:{json.dumps(result, sort_keys=True)}"
            return hashlib.sha256(log_string.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to generate log hash: {str(e)}")
            return "hash_generation_failed"

    def execute(self, state: MASXState) -> MASXState:
        """Execute the logging and auditing workflow."""
        try:
            self.logger.info("Starting logging auditor execution")

            # Get workflow information
            workflow_id = state.workflow.get("workflow_id", "unknown")
            start_time = state.workflow.get("start_time", datetime.utcnow().isoformat())

            # Monitor agent outputs
            agent_outputs = state.agents
            monitoring_results = self.monitor_agent_outputs(agent_outputs)

            # Log the overall workflow execution
            execution_time = (
                datetime.utcnow()
                - datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            ).total_seconds()

            workflow_log_id = self.log_agent_execution(
                agent_name="workflow_orchestrator",
                action="complete_workflow",
                parameters={
                    "workflow_id": workflow_id,
                    "agent_count": len(agent_outputs),
                },
                result={
                    "status": "completed",
                    "monitoring_results": monitoring_results,
                },
                execution_time=execution_time,
                status=(
                    "success"
                    if monitoring_results["overall_status"] == "healthy"
                    else "warning"
                ),
            )

            # Generate audit report
            audit_report = self.generate_audit_report(
                workflow_id=workflow_id,
                start_time=datetime.fromisoformat(start_time.replace("Z", "+00:00")),
                end_time=datetime.utcnow(),
            )

            # Update state
            state.agents[self.name] = {
                "status": "completed",
                "output": {
                    "workflow_log_id": workflow_log_id,
                    "monitoring_results": monitoring_results,
                    "audit_report": audit_report,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            self.logger.info("Logging auditor execution completed successfully")
            return state

        except Exception as e:
            self.logger.error(f"Logging auditor execution failed: {str(e)}")
            state.agents[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
            return state
