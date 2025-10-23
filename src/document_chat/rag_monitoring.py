"""
RAG Monitoring and Alerting System
Tracks retrieval quality, hallucination rates, and sends alerts for anomalies
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import json
import requests


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: str
    query: str
    timestamp: datetime
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    chunks_retrieved: int
    top_similarity_score: float
    avg_similarity_score: float
    answer_length: int
    hallucination_score: Optional[float] = None
    is_grounded: Optional[bool] = None
    user_feedback: Optional[str] = None  # "positive", "negative", None


@dataclass
class AlertThresholds:
    """Configurable alert thresholds"""
    hallucination_rate: float = 0.05  # 5%
    avg_retrieval_time_ms: float = 500.0
    min_similarity_score: float = 0.5
    error_rate: float = 0.1  # 10%
    low_confidence_rate: float = 0.3  # 30%


@dataclass
class Alert:
    """Alert notification"""
    severity: str  # "INFO", "WARNING", "CRITICAL"
    metric: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ALERT NOTIFIER (Abstract + Implementations)
# ============================================================================

class AlertNotifier:
    """Abstract base for alert notifications"""

    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert notification.
        Returns True if successful, False otherwise.
        """
        raise NotImplementedError


class SlackAlertNotifier(AlertNotifier):
    """Send alerts to Slack via webhook"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        if not self.webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return False

        # Format message with emoji based on severity
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🚨"
        }.get(alert.severity, "📊")

        message = {
            "text": f"{emoji} *RAG System Alert*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {alert.severity}: {alert.metric}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Message:*\n{alert.message}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Current Value:*\n{alert.current_value:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Threshold:*\n{alert.threshold_value:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Timestamp:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
        }

        # Add metadata if available
        if alert.metadata:
            metadata_text = "\n".join([f"• {k}: {v}" for k, v in alert.metadata.items()])
            message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Additional Details:*\n{metadata_text}"
                }
            })

        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=5
            )
            response.raise_for_status()
            self.logger.info(f"Alert sent to Slack: {alert.metric}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return False


class EmailAlertNotifier(AlertNotifier):
    """Send alerts via email (placeholder for SMTP integration)"""

    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        # TODO: Implement SMTP email sending
        self.logger.info(f"Email alert (placeholder): {alert.message}")
        return True


class PagerDutyAlertNotifier(AlertNotifier):
    """Send alerts to PagerDuty"""

    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to PagerDuty"""
        if not self.integration_key:
            self.logger.warning("PagerDuty integration key not configured")
            return False

        severity_map = {
            "INFO": "info",
            "WARNING": "warning",
            "CRITICAL": "critical"
        }

        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"RAG System Alert: {alert.metric}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "document_portal_rag",
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": {
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold_value,
                    "metadata": alert.metadata
                }
            }
        }

        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            self.logger.info(f"Alert sent to PagerDuty: {alert.metric}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {str(e)}")
            return False


class LogAlertNotifier(AlertNotifier):
    """Log alerts (default fallback)"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Log alert"""
        log_method = {
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "CRITICAL": self.logger.critical
        }.get(alert.severity, self.logger.info)

        log_method(
            f"RAG Alert: {alert.metric} | {alert.message} | "
            f"Current: {alert.current_value:.2f}, Threshold: {alert.threshold_value:.2f}"
        )
        return True


# ============================================================================
# RAG MONITORING SYSTEM
# ============================================================================

class RAGMonitoring:
    """
    Monitor RAG system health and performance.
    Tracks metrics, detects anomalies, and sends alerts.
    """

    def __init__(
        self,
        thresholds: AlertThresholds = None,
        notifiers: List[AlertNotifier] = None,
        window_size: int = 100
    ):
        """
        Args:
            thresholds: Alert threshold configuration
            notifiers: List of alert notifiers (Slack, email, etc.)
            window_size: Number of recent queries to keep in memory
        """
        self.thresholds = thresholds or AlertThresholds()
        self.notifiers = notifiers or [LogAlertNotifier()]
        self.logger = logging.getLogger(__name__)

        # Metrics storage (sliding window)
        self.recent_queries: deque = deque(maxlen=window_size)
        self.session_start = datetime.now()

        # Counters
        self.total_queries = 0
        self.total_errors = 0
        self.total_hallucinations = 0

    def log_query(
        self,
        query_id: str,
        query: str,
        retrieval_time_ms: float,
        generation_time_ms: float,
        chunks_retrieved: int,
        top_similarity_score: float,
        avg_similarity_score: float,
        answer_length: int,
        hallucination_score: Optional[float] = None,
        is_grounded: Optional[bool] = None
    ) -> None:
        """
        Log metrics for a single query.
        Automatically checks thresholds and sends alerts if needed.
        """
        metrics = QueryMetrics(
            query_id=query_id,
            query=query,
            timestamp=datetime.now(),
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=retrieval_time_ms + generation_time_ms,
            chunks_retrieved=chunks_retrieved,
            top_similarity_score=top_similarity_score,
            avg_similarity_score=avg_similarity_score,
            answer_length=answer_length,
            hallucination_score=hallucination_score,
            is_grounded=is_grounded
        )

        self.recent_queries.append(metrics)
        self.total_queries += 1

        if hallucination_score and hallucination_score > 0.5:
            self.total_hallucinations += 1

        # Check thresholds and send alerts
        self._check_thresholds(metrics)

        self.logger.info(
            f"Query logged: {query_id} | "
            f"Retrieval: {retrieval_time_ms:.0f}ms | "
            f"Similarity: {top_similarity_score:.2f} | "
            f"Grounded: {is_grounded}"
        )

    def log_error(self, query_id: str, error_type: str, error_message: str) -> None:
        """Log an error in the RAG pipeline"""
        self.total_errors += 1
        self.logger.error(f"RAG Error: {query_id} | {error_type} | {error_message}")

        # Check error rate
        error_rate = self.total_errors / max(self.total_queries, 1)
        if error_rate > self.thresholds.error_rate:
            self._send_alert(Alert(
                severity="CRITICAL",
                metric="Error Rate",
                message=f"Error rate exceeded threshold: {error_rate:.1%}",
                current_value=error_rate,
                threshold_value=self.thresholds.error_rate,
                timestamp=datetime.now(),
                metadata={"recent_error": error_message}
            ))

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for recent queries"""
        if not self.recent_queries:
            return {"message": "No queries logged yet"}

        queries = list(self.recent_queries)

        # Calculate statistics
        avg_retrieval_time = sum(q.retrieval_time_ms for q in queries) / len(queries)
        avg_generation_time = sum(q.generation_time_ms for q in queries) / len(queries)
        avg_total_time = sum(q.total_time_ms for q in queries) / len(queries)
        avg_similarity = sum(q.avg_similarity_score for q in queries) / len(queries)

        # Hallucination metrics
        grounded_queries = [q for q in queries if q.is_grounded is not None]
        hallucination_rate = (
            len([q for q in grounded_queries if not q.is_grounded]) / len(grounded_queries)
            if grounded_queries else 0.0
        )

        return {
            "session_duration": str(datetime.now() - self.session_start),
            "total_queries": self.total_queries,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_queries, 1),
            "hallucination_rate": hallucination_rate,
            "avg_retrieval_time_ms": avg_retrieval_time,
            "avg_generation_time_ms": avg_generation_time,
            "avg_total_time_ms": avg_total_time,
            "avg_similarity_score": avg_similarity,
            "recent_query_count": len(queries)
        }

    def _check_thresholds(self, metrics: QueryMetrics) -> None:
        """Check if metrics exceed alert thresholds"""
        now = datetime.now()

        # Check retrieval time
        if metrics.retrieval_time_ms > self.thresholds.avg_retrieval_time_ms * 2:
            self._send_alert(Alert(
                severity="WARNING",
                metric="Retrieval Time",
                message=f"Slow retrieval detected: {metrics.retrieval_time_ms:.0f}ms",
                current_value=metrics.retrieval_time_ms,
                threshold_value=self.thresholds.avg_retrieval_time_ms,
                timestamp=now,
                metadata={"query": metrics.query[:100]}
            ))

        # Check similarity score
        if metrics.top_similarity_score < self.thresholds.min_similarity_score:
            self._send_alert(Alert(
                severity="INFO",
                metric="Low Similarity Score",
                message=f"Low relevance detected: {metrics.top_similarity_score:.2f}",
                current_value=metrics.top_similarity_score,
                threshold_value=self.thresholds.min_similarity_score,
                timestamp=now,
                metadata={"query": metrics.query[:100]}
            ))

        # Check hallucination rate (rolling window)
        if len(self.recent_queries) >= 10:
            recent_with_scores = [
                q for q in list(self.recent_queries)[-20:]
                if q.hallucination_score is not None
            ]
            if recent_with_scores:
                recent_hallucination_rate = (
                    len([q for q in recent_with_scores if q.hallucination_score > 0.5])
                    / len(recent_with_scores)
                )

                if recent_hallucination_rate > self.thresholds.hallucination_rate:
                    self._send_alert(Alert(
                        severity="CRITICAL",
                        metric="Hallucination Rate",
                        message=f"High hallucination rate: {recent_hallucination_rate:.1%}",
                        current_value=recent_hallucination_rate,
                        threshold_value=self.thresholds.hallucination_rate,
                        timestamp=now,
                        metadata={
                            "window_size": len(recent_with_scores),
                            "hallucinated_queries": len([q for q in recent_with_scores if q.hallucination_score > 0.5])
                        }
                    ))

    def _send_alert(self, alert: Alert) -> None:
        """Send alert through all configured notifiers"""
        self.logger.warning(f"Sending alert: {alert.severity} - {alert.metric}")

        for notifier in self.notifiers:
            try:
                success = notifier.send_alert(alert)
                if not success:
                    self.logger.warning(f"Alert notifier {notifier.__class__.__name__} failed")
            except Exception as e:
                self.logger.error(f"Error sending alert via {notifier.__class__.__name__}: {str(e)}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import os

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create monitoring system with multiple notifiers
    notifiers = [
        LogAlertNotifier(),  # Always log
    ]

    # Add Slack if webhook is configured
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        notifiers.append(SlackAlertNotifier(slack_webhook))

    # Add PagerDuty if configured
    pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
    if pagerduty_key:
        notifiers.append(PagerDutyAlertNotifier(pagerduty_key))

    # Initialize monitoring
    monitoring = RAGMonitoring(
        thresholds=AlertThresholds(
            hallucination_rate=0.05,
            avg_retrieval_time_ms=500.0,
            min_similarity_score=0.5
        ),
        notifiers=notifiers
    )

    # Simulate logging queries
    monitoring.log_query(
        query_id="q_001",
        query="What is the company revenue?",
        retrieval_time_ms=250.0,
        generation_time_ms=1500.0,
        chunks_retrieved=5,
        top_similarity_score=0.85,
        avg_similarity_score=0.75,
        answer_length=150,
        hallucination_score=0.1,
        is_grounded=True
    )

    # Simulate slow query (triggers alert)
    monitoring.log_query(
        query_id="q_002",
        query="Who is the CEO?",
        retrieval_time_ms=1200.0,  # Slow!
        generation_time_ms=800.0,
        chunks_retrieved=3,
        top_similarity_score=0.65,
        avg_similarity_score=0.55,
        answer_length=80,
        hallucination_score=0.3,
        is_grounded=True
    )

    # Get summary
    print("\n=== Summary Stats ===")
    print(json.dumps(monitoring.get_summary_stats(), indent=2))
