# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

"""
Unit tests for parallel execution utilities.

Tests parallel task execution, coordination, error handling,
and performance monitoring capabilities.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from app.workflows.parallel import (
    ParallelTaskExecutor,
    ParallelTask,
    TaskResult,
    TaskStatus,
    execute_parallel_tasks,
    create_task,
    TaskCoordinator,
    execute_agent_tasks,
    execute_data_fetching_tasks,
)
from app.core.exceptions import WorkflowException


class TestParallelTask:
    """Test cases for ParallelTask dataclass."""

    def test_parallel_task_creation(self):
        """Test ParallelTask creation with default values."""

        def test_func():
            return "test"

        task = ParallelTask(id="test_task", func=test_func)

        assert task.id == "test_task"
        assert task.func == test_func
        assert task.args == ()
        assert task.kwargs == {}
        assert task.timeout is None
        assert task.retries == 0
        assert task.priority == 0

    def test_parallel_task_creation_with_all_params(self):
        """Test ParallelTask creation with all parameters."""

        def test_func(arg1, arg2, kwarg1=None):
            return f"{arg1}_{arg2}_{kwarg1}"

        task = ParallelTask(
            id="test_task",
            func=test_func,
            args=("value1", "value2"),
            kwargs={"kwarg1": "kwvalue1"},
            timeout=30.0,
            retries=3,
            priority=5,
        )

        assert task.id == "test_task"
        assert task.func == test_func
        assert task.args == ("value1", "value2")
        assert task.kwargs == {"kwarg1": "kwvalue1"}
        assert task.timeout == 30.0
        assert task.retries == 3
        assert task.priority == 5


class TestTaskResult:
    """Test cases for TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test TaskResult creation with default values."""
        result = TaskResult(task_id="test_task", status=TaskStatus.COMPLETED)

        assert result.task_id == "test_task"
        assert result.status == TaskStatus.COMPLETED
        assert result.result is None
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.retry_count == 0

    def test_task_result_creation_with_all_params(self):
        """Test TaskResult creation with all parameters."""
        result = TaskResult(
            task_id="test_task",
            status=TaskStatus.FAILED,
            result={"data": "test"},
            error="Test error",
            execution_time=1.5,
            retry_count=2,
        )

        assert result.task_id == "test_task"
        assert result.status == TaskStatus.FAILED
        assert result.result == {"data": "test"}
        assert result.error == "Test error"
        assert result.execution_time == 1.5
        assert result.retry_count == 2


class TestParallelTaskExecutor:
    """Test cases for ParallelTaskExecutor."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.workflows.parallel.get_workflow_logger"):
            self.executor = ParallelTaskExecutor(
                max_concurrent=3, max_retries=2, timeout=60.0
            )

    def test_initialization(self):
        """Test executor initialization."""
        assert self.executor.max_concurrent == 3
        assert self.executor.max_retries == 2
        assert self.executor.default_timeout == 60.0
        assert self.executor.enable_monitoring is True
        assert self.executor.total_tasks_executed == 0
        assert self.executor.total_execution_time == 0.0
        assert self.executor.failed_tasks == 0
        assert self.executor.retry_count == 0

    def test_execute_empty_tasks(self):
        """Test execution with empty task list."""

        async def test_async():
            result = await self.executor.execute([])
            return result

        result = asyncio.run(test_async())
        assert result == []

    def test_execute_single_task_success(self):
        """Test successful execution of a single task."""

        def test_func():
            return "success"

        task = ParallelTask(id="test_task", func=test_func)

        async def test_async():
            result = await self.executor.execute([task])
            return result

        result = asyncio.run(test_async())

        assert len(result) == 1
        assert result[0].task_id == "test_task"
        assert result[0].status == TaskStatus.COMPLETED
        assert result[0].result == "success"
        assert result[0].error is None
        assert result[0].execution_time > 0
        assert result[0].retry_count == 0

    def test_execute_multiple_tasks_success(self):
        """Test successful execution of multiple tasks."""

        def task1():
            return "result1"

        def task2():
            return "result2"

        tasks = [
            ParallelTask(id="task1", func=task1),
            ParallelTask(id="task2", func=task2),
        ]

        async def test_async():
            result = await self.executor.execute(tasks)
            return result

        result = asyncio.run(test_async())

        assert len(result) == 2
        assert result[0].status == TaskStatus.COMPLETED
        assert result[1].status == TaskStatus.COMPLETED
        assert result[0].result == "result1"
        assert result[1].result == "result2"

    def test_execute_task_with_timeout(self):
        """Test task execution with timeout."""

        def slow_func():
            time.sleep(2)  # Simulate slow operation
            return "slow_result"

        task = ParallelTask(
            id="slow_task", func=slow_func, timeout=1.0, retries=1  # 1 second timeout
        )

        async def test_async():
            result = await self.executor.execute([task])
            return result

        result = asyncio.run(test_async())

        assert len(result) == 1
        assert result[0].status == TaskStatus.FAILED
        assert "timed out" in result[0].error
        assert result[0].retry_count > 0

    def test_execute_task_with_retries(self):
        """Test task execution with retries."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success_after_retries"

        task = ParallelTask(id="retry_task", func=failing_func, retries=3)

        async def test_async():
            result = await self.executor.execute([task])
            return result

        result = asyncio.run(test_async())

        assert len(result) == 1
        assert result[0].status == TaskStatus.COMPLETED
        assert result[0].result == "success_after_retries"
        assert result[0].retry_count == 2  # Failed twice, succeeded on third try
        assert call_count == 3

    def test_execute_task_with_priority(self):
        """Test task execution with priority ordering."""
        execution_order = []

        def low_priority():
            execution_order.append("low")
            return "low_result"

        def high_priority():
            execution_order.append("high")
            return "high_result"

        tasks = [
            ParallelTask(id="low", func=low_priority, priority=1),
            ParallelTask(id="high", func=high_priority, priority=10),
        ]

        async def test_async():
            result = await self.executor.execute(tasks)
            return result

        result = asyncio.run(test_async())

        assert len(result) == 2
        assert result[0].status == TaskStatus.COMPLETED
        assert result[1].status == TaskStatus.COMPLETED
        # High priority task should be executed first
        assert execution_order[0] == "high"
        assert execution_order[1] == "low"

    def test_execute_async_task(self):
        """Test execution of async tasks."""

        async def async_func():
            await asyncio.sleep(0.1)
            return "async_result"

        task = ParallelTask(id="async_task", func=async_func)

        async def test_async():
            result = await self.executor.execute([task])
            return result

        result = asyncio.run(test_async())

        assert len(result) == 1
        assert result[0].status == TaskStatus.COMPLETED
        assert result[0].result == "async_result"

    def test_execute_task_with_exception(self):
        """Test task execution that raises an exception."""

        def failing_func():
            raise ValueError("Test error")

        task = ParallelTask(id="failing_task", func=failing_func, retries=1)

        async def test_async():
            result = await self.executor.execute([task])
            return result

        result = asyncio.run(test_async())

        assert len(result) == 1
        assert result[0].status == TaskStatus.FAILED
        assert "Test error" in result[0].error
        assert result[0].retry_count > 0

    def test_get_performance_stats(self):
        """Test performance statistics calculation."""

        def test_func():
            return "test"

        task = ParallelTask(id="test_task", func=test_func)

        async def test_async():
            await self.executor.execute([task])
            return self.executor.get_performance_stats()

        stats = asyncio.run(test_async())

        assert stats["total_tasks_executed"] == 1
        assert stats["total_execution_time"] > 0
        assert stats["average_execution_time"] > 0
        assert stats["failed_tasks"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["retry_count"] == 0
        assert stats["active_tasks"] == 0
        assert stats["max_concurrent"] == 3


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_create_task(self):
        """Test create_task utility function."""

        def test_func(arg1, arg2, kwarg1=None):
            return f"{arg1}_{arg2}_{kwarg1}"

        task = create_task(
            task_id="test_task",
            func=test_func,
            timeout=30.0,
            retries=2,
            priority=5,
            kwarg1="kwvalue1",
        )

        assert task.id == "test_task"
        assert task.func == test_func
        assert task.args == ("value1", "value2")
        assert task.kwargs == {"kwarg1": "kwvalue1"}
        assert task.timeout == 30.0
        assert task.retries == 2
        assert task.priority == 5

    def test_execute_parallel_tasks(self):
        """Test execute_parallel_tasks utility function."""

        def task1():
            return "result1"

        def task2():
            return "result2"

        tasks = [
            ParallelTask(id="task1", func=task1),
            ParallelTask(id="task2", func=task2),
        ]

        async def test_async():
            result = await execute_parallel_tasks(tasks, max_concurrent=2)
            return result

        result = asyncio.run(test_async())

        assert len(result) == 2
        assert result[0].status == TaskStatus.COMPLETED
        assert result[1].status == TaskStatus.COMPLETED
        assert result[0].result == "result1"
        assert result[1].result == "result2"


class TestTaskCoordinator:
    """Test cases for TaskCoordinator."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.workflows.parallel.get_workflow_logger"):
            self.coordinator = TaskCoordinator(max_concurrent=3)

    def test_initialization(self):
        """Test coordinator initialization."""
        assert self.coordinator.max_concurrent == 3
        assert self.coordinator.executor is not None
        assert self.coordinator._task_dependencies == {}
        assert self.coordinator._task_results == {}

    def test_add_task_dependency(self):
        """Test adding task dependencies."""
        self.coordinator.add_task_dependency("task2", ["task1"])
        self.coordinator.add_task_dependency("task3", ["task1", "task2"])

        assert self.coordinator._task_dependencies["task2"] == ["task1"]
        assert self.coordinator._task_dependencies["task3"] == ["task1", "task2"]

    def test_create_execution_phases_simple(self):
        """Test creation of execution phases for simple dependencies."""

        def task1():
            return "result1"

        def task2():
            return "result2"

        def task3():
            return "result3"

        tasks = [
            ParallelTask(id="task1", func=task1),
            ParallelTask(id="task2", func=task2),
            ParallelTask(id="task3", func=task3),
        ]

        # task2 depends on task1, task3 depends on task2
        self.coordinator.add_task_dependency("task2", ["task1"])
        self.coordinator.add_task_dependency("task3", ["task2"])

        phases = self.coordinator._create_execution_phases(tasks)

        assert len(phases) == 3
        assert len(phases[0]) == 1  # task1 only
        assert len(phases[1]) == 1  # task2 only
        assert len(phases[2]) == 1  # task3 only
        assert phases[0][0].id == "task1"
        assert phases[1][0].id == "task2"
        assert phases[2][0].id == "task3"

    def test_create_execution_phases_parallel(self):
        """Test creation of execution phases for parallel tasks."""

        def task1():
            return "result1"

        def task2():
            return "result2"

        def task3():
            return "result3"

        tasks = [
            ParallelTask(id="task1", func=task1),
            ParallelTask(id="task2", func=task2),
            ParallelTask(id="task3", func=task3),
        ]

        # task3 depends on task1 and task2, but task1 and task2 are independent
        self.coordinator.add_task_dependency("task3", ["task1", "task2"])

        phases = self.coordinator._create_execution_phases(tasks)

        assert len(phases) == 2
        assert len(phases[0]) == 2  # task1 and task2 can run in parallel
        assert len(phases[1]) == 1  # task3 runs after both complete
        assert phases[1][0].id == "task3"

    def test_execute_with_dependencies(self):
        """Test execution with dependencies."""

        def task1():
            return "result1"

        def task2():
            return "result2"

        tasks = [
            ParallelTask(id="task1", func=task1),
            ParallelTask(id="task2", func=task2),
        ]

        dependencies = {"task2": ["task1"]}

        async def test_async():
            result = await self.coordinator.execute_with_dependencies(
                tasks, dependencies
            )
            return result

        result = asyncio.run(test_async())

        assert len(result) == 2
        assert "task1" in result
        assert "task2" in result
        assert result["task1"].status == TaskStatus.COMPLETED
        assert result["task2"].status == TaskStatus.COMPLETED

    def test_get_task_result(self):
        """Test getting task result."""
        result = TaskResult(
            task_id="test_task", status=TaskStatus.COMPLETED, result="test_result"
        )

        self.coordinator._task_results["test_task"] = result

        retrieved_result = self.coordinator.get_task_result("test_task")
        assert retrieved_result == result

        # Test non-existent task
        assert self.coordinator.get_task_result("non_existent") is None

    def test_get_failed_tasks(self):
        """Test getting failed tasks."""
        success_result = TaskResult(task_id="success_task", status=TaskStatus.COMPLETED)

        failed_result = TaskResult(
            task_id="failed_task", status=TaskStatus.FAILED, error="Test error"
        )

        self.coordinator._task_results["success_task"] = success_result
        self.coordinator._task_results["failed_task"] = failed_result

        failed_tasks = self.coordinator.get_failed_tasks()

        assert len(failed_tasks) == 1
        assert failed_tasks[0].task_id == "failed_task"
        assert failed_tasks[0].status == TaskStatus.FAILED


class TestAgentExecution:
    """Test cases for agent execution utilities."""

    def test_execute_agent_tasks(self):
        """Test execute_agent_tasks utility function."""
        # Mock agents
        mock_agent1 = Mock()
        mock_agent1.name = "agent1"
        mock_agent1.run.return_value = {"result": "agent1_result"}

        mock_agent2 = Mock()
        mock_agent2.name = "agent2"
        mock_agent2.run.return_value = {"result": "agent2_result"}

        agents = [mock_agent1, mock_agent2]
        input_data = {"test": "data"}

        async def test_async():
            result = await execute_agent_tasks(agents, input_data, max_concurrent=2)
            return result

        result = asyncio.run(test_async())

        assert len(result) == 2
        assert result["agent1"]["result"] == "agent1_result"
        assert result["agent2"]["result"] == "agent2_result"

        # Verify agents were called with correct input
        mock_agent1.run.assert_called_once_with(input_data)
        mock_agent2.run.assert_called_once_with(input_data)

    def test_execute_data_fetching_tasks(self):
        """Test execute_data_fetching_tasks utility function."""
        # Mock fetchers
        mock_fetcher1 = Mock()
        mock_fetcher1.name = "fetcher1"
        mock_fetcher1.fetch.return_value = ["data1", "data2"]

        mock_fetcher2 = Mock()
        mock_fetcher2.name = "fetcher2"
        mock_fetcher2.fetch.return_value = ["data3", "data4"]

        fetchers = [mock_fetcher1, mock_fetcher2]
        queries = ["query1", "query2"]

        async def test_async():
            result = await execute_data_fetching_tasks(
                fetchers, queries, max_concurrent=2
            )
            return result

        result = asyncio.run(test_async())

        assert len(result) == 2
        assert result["fetcher1"] == ["data1", "data2"]
        assert result["fetcher2"] == ["data3", "data4"]

        # Verify fetchers were called with correct queries
        mock_fetcher1.fetch.assert_called_once_with(queries)
        mock_fetcher2.fetch.assert_called_once_with(queries)
