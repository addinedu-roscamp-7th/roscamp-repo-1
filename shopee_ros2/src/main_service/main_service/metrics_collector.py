"""
메트릭 수집기

시스템 전체의 성능 메트릭, 에러 통계, 네트워크 상태 등을 수집한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    시스템 전체 메트릭 수집기

    OrderService, ProductService, APIController 등으로부터
    통계 정보를 수집하여 대시보드에 제공한다.
    """

    def __init__(
        self,
        order_service=None,
        product_service=None,
        api_controller=None,
        robot_coordinator=None,
        database_manager=None,
    ):
        self._order_service = order_service
        self._product_service = product_service
        self._api_controller = api_controller
        self._robot_coordinator = robot_coordinator
        self._database_manager = database_manager

    async def collect_metrics(self, robot_states: list) -> Dict[str, Any]:
        """
        전체 시스템 메트릭을 수집한다.

        Args:
            robot_states: 현재 로봇 상태 목록

        Returns:
            메트릭 딕셔너리
        """
        metrics = {}

        # 성능 메트릭
        if self._order_service:
            performance = await self._get_performance_metrics()
            metrics.update(performance)

        # 에러 및 장애 추적
        if self._order_service:
            error_stats = await self._get_error_stats()
            metrics['failed_orders'] = error_stats.get('failed_orders', [])
            metrics['failed_orders_by_reason'] = error_stats.get('by_reason', {})

        # 로봇 오류 현황
        error_robots = [r for r in robot_states if r.get('status') == 'ERROR']
        offline_robots = [r for r in robot_states if r.get('status') == 'OFFLINE']
        metrics['error_robots'] = error_robots
        metrics['offline_robots'] = offline_robots

        # LLM 서비스 상태
        if self._product_service and hasattr(self._product_service, 'get_llm_stats'):
            metrics['llm_stats'] = self._product_service.get_llm_stats()
        elif self._product_service and hasattr(self._product_service, '_llm_client'):
            # LLMClient에서 직접 통계 가져오기
            try:
                llm_client = self._product_service._llm_client
                if hasattr(llm_client, 'get_stats_snapshot'):
                    metrics['llm_stats'] = llm_client.get_stats_snapshot()
                else:
                    metrics['llm_stats'] = {
                        'success_rate': 0.0,
                        'avg_response_time': 0,
                        'fallback_count': 0,
                        'failure_count': 0,
                        'success_count': 0,
                    }
            except Exception as e:
                logger.error(f'Failed to get LLM stats: {e}')
                metrics['llm_stats'] = {
                    'success_rate': 0.0,
                    'avg_response_time': 0,
                    'fallback_count': 0,
                    'failure_count': 0,
                    'success_count': 0,
                }
        else:
            metrics['llm_stats'] = {
                'success_rate': 0.0,
                'avg_response_time': 0,
                'fallback_count': 0,
                'failure_count': 0,
                'success_count': 0,
            }

        # ROS 재시도 통계
        if self._robot_coordinator and hasattr(self._robot_coordinator, 'get_retry_stats'):
            metrics['ros_retry_count'] = self._robot_coordinator.get_retry_stats()
        else:
            metrics['ros_retry_count'] = 0

        # 네트워크 상태
        network_stats = await self._get_network_stats()
        metrics['network'] = network_stats

        return metrics

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """
        시스템 성능 메트릭을 계산한다.

        Returns:
            평균 처리 시간, 처리량, 성공률 등
        """
        try:
            # 기본값 설정
            metrics = {
                'avg_processing_time': 0.0,
                'hourly_throughput': 0,
                'success_rate': 0.0,
                'robot_utilization': 0.0,
                'system_load': 0.0,
                'active_orders': 0,
            }

            # OrderService가 get_performance_metrics를 지원하는 경우
            if hasattr(self._order_service, 'get_performance_metrics'):
                try:
                    perf = await self._order_service.get_performance_metrics()
                    metrics.update(perf)
                except Exception as e:
                    logger.warning(f'Failed to get performance metrics from OrderService: {e}')
            else:
                logger.debug('OrderService.get_performance_metrics() not available')

            return metrics

        except Exception as e:
            logger.error(f'Failed to collect performance metrics: {e}')
            return {
                'avg_processing_time': 0.0,
                'hourly_throughput': 0,
                'success_rate': 0.0,
                'robot_utilization': 0.0,
                'system_load': 0.0,
                'active_orders': 0,
            }

    async def _get_error_stats(self) -> Dict[str, Any]:
        """
        에러 통계를 수집한다.

        Returns:
            실패한 주문 목록 및 사유별 집계
        """
        try:
            error_stats: Dict[str, Any] = {
                'failed_orders': [],
                'by_reason': {},
            }

            # OrderService가 get_recent_failed_orders를 지원하는 경우
            if hasattr(self._order_service, 'get_recent_failed_orders'):
                try:
                    # 최근 실패 주문 조회 (limit만 사용)
                    failed_orders = await self._order_service.get_recent_failed_orders(
                        limit=10
                    )
                    error_stats['failed_orders'] = failed_orders

                    # 사유별 집계
                    by_reason: Dict[str, int] = {}
                    for order in failed_orders:
                        reason = order.get('failure_reason', 'UNKNOWN')
                        by_reason[reason] = by_reason.get(reason, 0) + 1
                    error_stats['by_reason'] = by_reason
                except Exception as e:
                    logger.warning(f'Failed to get recent failed orders: {e}')
            
            
            # 사유별 집계 (get_failed_orders_by_reason 메서드 사용)
            if hasattr(self._order_service, 'get_failed_orders_by_reason'):
                try:
                    by_reason = await self._order_service.get_failed_orders_by_reason(
                        window_minutes=60
                    )
                    error_stats['by_reason'] = by_reason
                except Exception as e:
                    logger.warning(f'Failed to get failed orders by reason: {e}')
            
            if not error_stats['failed_orders'] and not error_stats['by_reason']:
                logger.debug('OrderService error statistics methods not available')

            return error_stats

        except Exception as e:
            logger.error(f'Failed to collect error stats: {e}')
            return {
                'failed_orders': [],
                'by_reason': {},
            }

    async def _get_network_stats(self) -> Dict[str, Any]:
        """
        네트워크 및 연결 상태를 수집한다.

        Returns:
            App 세션, ROS 토픽, DB 커넥션 상태
        """
        network = {
            'app_sessions': 0,
            'app_sessions_max': 200,
            'ros_topics_healthy': True,
            'ros_topic_health': {},
            'topic_receive_rate': 100.0,
            'event_topic_activity': {},
            'event_topic_timeout': None,
            'llm_response_time': 0.0,
            'db_connections': 0,
            'db_connections_max': 10,
        }

        try:
            # App 연결 세션 수
            if self._api_controller and hasattr(self._api_controller, 'get_connection_stats'):
                conn_stats = self._api_controller.get_connection_stats()
                network['app_sessions'] = conn_stats.get('app_sessions', 0)
                network['app_sessions_max'] = conn_stats.get('app_sessions_max', 200)
            elif self._api_controller and hasattr(self._api_controller, '_active_connections'):
                network['app_sessions'] = len(self._api_controller._active_connections)

            # ROS 토픽 헬스
            if self._robot_coordinator and hasattr(self._robot_coordinator, 'get_topic_health'):
                try:
                    topic_health = self._robot_coordinator.get_topic_health()
                    if topic_health:
                        network['ros_topics_healthy'] = topic_health.get('ros_topics_healthy', True)
                        network['ros_topic_health'] = topic_health.get('ros_topic_health', {})
                        network['topic_receive_rate'] = topic_health.get('topic_receive_rate', 100.0)
                        network['event_topic_activity'] = topic_health.get('event_topic_activity', {})
                        network['event_topic_timeout'] = topic_health.get('event_topic_timeout')
                except Exception as e:
                    logger.debug(f'Failed to get topic health: {e}')

            # DB 커넥션 풀
            if self._database_manager and hasattr(self._database_manager, 'get_pool_stats'):
                try:
                    pool_stats = self._database_manager.get_pool_stats()
                    network['db_connections'] = pool_stats.get('db_connections', 0)
                    network['db_connections_max'] = pool_stats.get('db_connections_max', 10)
                except Exception as e:
                    logger.debug(f'Failed to get DB pool stats: {e}')

            # LLM 응답 시간 (product_service의 _llm_client에서 가져오기)
            if self._product_service and hasattr(self._product_service, '_llm_client'):
                try:
                    llm_client = self._product_service._llm_client
                    if hasattr(llm_client, 'get_stats_snapshot'):
                        llm_stats = llm_client.get_stats_snapshot()
                        network['llm_response_time'] = llm_stats.get('avg_response_time', 0.0)
                except Exception as e:
                    logger.debug(f'Failed to get LLM response time: {e}')

        except Exception as e:
            logger.error(f'Failed to collect network stats: {e}')

        return network
