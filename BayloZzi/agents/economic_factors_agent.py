# agents/economic_factors_agent.py

import asyncio
import logging
import numpy as np
import pandas as pd
import requests
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
import yfinance as yf
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EconomicImpact(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    SLIGHTLY_POSITIVE = "slightly_positive"
    NEUTRAL = "neutral"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class IndicatorImportance(Enum):
    CRITICAL = "critical"     # GDP, Interest Rates, Employment
    HIGH = "high"            # Inflation, Trade Balance
    MEDIUM = "medium"        # Manufacturing, Consumer Confidence
    LOW = "low"              # Minor indicators

@dataclass
class EconomicIndicator:
    name: str
    country: str
    value: float
    previous_value: float
    forecast: float
    release_date: datetime
    importance: IndicatorImportance
    impact: EconomicImpact
    surprise_factor: float  # How much it deviated from forecast
    historical_correlation: float  # Correlation with currency
    market_reaction_potential: float

@dataclass
class CentralBankPolicy:
    bank: str
    country: str
    current_rate: float
    previous_rate: float
    policy_stance: str  # hawkish, dovish, neutral
    next_meeting: datetime
    rate_change_probability: Dict[str, float]  # probability of hike/hold/cut
    forward_guidance: str
    impact_assessment: EconomicImpact

@dataclass
class EconomicAnalysis:
    country: str
    currency: str
    overall_health: EconomicImpact
    gdp_growth_trend: str
    inflation_trend: str
    employment_trend: str
    monetary_policy_bias: str
    fiscal_policy_impact: EconomicImpact
    trade_balance_impact: EconomicImpact
    key_indicators: List[EconomicIndicator]
    central_bank_policy: CentralBankPolicy
    risk_factors: List[str]
    opportunities: List[str]
    currency_outlook: Dict[str, float]  # 1m, 3m, 6m outlook

class EconomicFactorsAgent:
    """
    Advanced Economic Factors Analysis Agent for Multi-Agent Forex Trading System.
    Analyzes macroeconomic indicators and their impact on currency valuations.
    """
    
    def __init__(self, agent_id: str = "economic_factors"):
        self.agent_id = agent_id
        
        # Redis connection for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # API configurations
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.trading_economics_key = os.getenv("TRADING_ECONOMICS_KEY")
        
        # Initialize FRED API if available
        self.fred = None
        if self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
            except:
                logger.warning("FRED API initialization failed")
        
        # Economic indicator mapping
        self.economic_indicators = {
            'USD': {
                'gdp': {'fred_id': 'GDP', 'importance': IndicatorImportance.CRITICAL},
                'unemployment': {'fred_id': 'UNRATE', 'importance': IndicatorImportance.CRITICAL},
                'inflation': {'fred_id': 'CPIAUCSL', 'importance': IndicatorImportance.HIGH},
                'fed_rate': {'fred_id': 'FEDFUNDS', 'importance': IndicatorImportance.CRITICAL},
                'nonfarm_payrolls': {'fred_id': 'PAYEMS', 'importance': IndicatorImportance.CRITICAL},
                'trade_balance': {'fred_id': 'BOPGSTB', 'importance': IndicatorImportance.HIGH},
                'retail_sales': {'fred_id': 'RSAFS', 'importance': IndicatorImportance.MEDIUM},
                'manufacturing_pmi': {'fred_id': 'MANEMP', 'importance': IndicatorImportance.MEDIUM}
            },
            'EUR': {
                'gdp': {'symbol': 'EU_GDP', 'importance': IndicatorImportance.CRITICAL},
                'unemployment': {'symbol': 'EU_UNEMPLOYMENT', 'importance': IndicatorImportance.HIGH},
                'inflation': {'symbol': 'EU_CPI', 'importance': IndicatorImportance.HIGH},
                'ecb_rate': {'symbol': 'ECB_RATE', 'importance': IndicatorImportance.CRITICAL}
            },
            'GBP': {
                'gdp': {'symbol': 'UK_GDP', 'importance': IndicatorImportance.CRITICAL},
                'unemployment': {'symbol': 'UK_UNEMPLOYMENT', 'importance': IndicatorImportance.HIGH},
                'inflation': {'symbol': 'UK_CPI', 'importance': IndicatorImportance.HIGH},
                'boe_rate': {'symbol': 'BOE_RATE', 'importance': IndicatorImportance.CRITICAL}
            },
            'JPY': {
                'gdp': {'symbol': 'JP_GDP', 'importance': IndicatorImportance.CRITICAL},
                'unemployment': {'symbol': 'JP_UNEMPLOYMENT', 'importance': IndicatorImportance.HIGH},
                'inflation': {'symbol': 'JP_CPI', 'importance': IndicatorImportance.HIGH},
                'boj_rate': {'symbol': 'BOJ_RATE', 'importance': IndicatorImportance.CRITICAL}
            }
        }
        
        # Central bank information
        self.central_banks = {
            'USD': {
                'name': 'Federal Reserve',
                'chair': 'Jerome Powell',
                'meeting_frequency': 8,  # per year
                'typical_meeting_months': [1, 3, 5, 6, 7, 9, 11, 12]
            },
            'EUR': {
                'name': 'European Central Bank',
                'chair': 'Christine Lagarde',
                'meeting_frequency': 8,
                'typical_meeting_months': [1, 2, 3, 4, 6, 7, 9, 10, 12]
            },
            'GBP': {
                'name': 'Bank of England',
                'chair': 'Andrew Bailey',
                'meeting_frequency': 8,
                'typical_meeting_months': [2, 3, 5, 6, 8, 9, 11, 12]
            },
            'JPY': {
                'name': 'Bank of Japan',
                'chair': 'Kazuo Ueda',
                'meeting_frequency': 8,
                'typical_meeting_months': [1, 3, 4, 6, 7, 9, 10, 12]
            }
        }
        
        # Historical correlations (will be updated dynamically)
        self.indicator_correlations = {
            'USD': {
                'gdp': 0.7, 'unemployment': -0.6, 'inflation': 0.4, 'fed_rate': 0.8,
                'nonfarm_payrolls': 0.75, 'trade_balance': 0.3
            },
            'EUR': {
                'gdp': 0.65, 'unemployment': -0.5, 'inflation': 0.35, 'ecb_rate': 0.7
            },
            'GBP': {
                'gdp': 0.6, 'unemployment': -0.55, 'inflation': 0.4, 'boe_rate': 0.75
            },
            'JPY': {
                'gdp': 0.5, 'unemployment': -0.4, 'inflation': 0.3, 'boj_rate': 0.6
            }
        }
        
        # Economic data cache
        self.economic_data_cache = {}
        self.analysis_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'accurate_predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0
        }
        
        # Market reaction patterns
        self.market_reaction_patterns = {
            'surprise_threshold': 0.2,  # 20% deviation from forecast
            'high_impact_threshold': 0.5,
            'correlation_threshold': 0.6
        }
    
    async def analyze_economic_factors(self, currency: str) -> EconomicAnalysis:
        """
        Comprehensive economic analysis for a specific currency.
        
        Args:
            currency: Currency code (USD, EUR, GBP, JPY, etc.)
            
        Returns:
            Complete economic analysis
        """
        try:
            country = self._get_country_from_currency(currency)
            
            # Fetch latest economic indicators
            indicators = await self._fetch_economic_indicators(currency)
            
            # Analyze central bank policy
            cb_policy = await self._analyze_central_bank_policy(currency)
            
            # Calculate overall economic health
            overall_health = self._calculate_economic_health(indicators, cb_policy)
            
            # Analyze trends
            gdp_trend = self._analyze_gdp_trend(indicators)
            inflation_trend = self._analyze_inflation_trend(indicators)
            employment_trend = self._analyze_employment_trend(indicators)
            
            # Determine monetary policy bias
            monetary_bias = self._determine_monetary_policy_bias(cb_policy, indicators)
            
            # Assess fiscal policy impact
            fiscal_impact = self._assess_fiscal_policy_impact(currency, indicators)
            
            # Analyze trade balance impact
            trade_impact = self._analyze_trade_balance_impact(currency, indicators)
            
            # Identify risk factors and opportunities
            risk_factors = self._identify_risk_factors(currency, indicators, cb_policy)
            opportunities = self._identify_opportunities(currency, indicators, cb_policy)
            
            # Generate currency outlook
            currency_outlook = self._generate_currency_outlook(
                currency, indicators, cb_policy, overall_health
            )
            
            analysis = EconomicAnalysis(
                country=country,
                currency=currency,
                overall_health=overall_health,
                gdp_growth_trend=gdp_trend,
                inflation_trend=inflation_trend,
                employment_trend=employment_trend,
                monetary_policy_bias=monetary_bias,
                fiscal_policy_impact=fiscal_impact,
                trade_balance_impact=trade_impact,
                key_indicators=indicators,
                central_bank_policy=cb_policy,
                risk_factors=risk_factors,
                opportunities=opportunities,
                currency_outlook=currency_outlook
            )
            
            # Store analysis
            self.analysis_history.append({
                'currency': currency,
                'timestamp': datetime.now(),
                'analysis': analysis
            })
            
            # Send to other agents if significant
            if overall_health in [EconomicImpact.VERY_POSITIVE, EconomicImpact.VERY_NEGATIVE] or \
               cb_policy.policy_stance in ['hawkish', 'dovish']:
                await self._send_economic_signal(currency, analysis)
            
            logger.info(f"Economic analysis completed for {currency}: {overall_health.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in economic analysis for {currency}: {str(e)}")
            return self._create_neutral_analysis(currency)
    
    def _get_country_from_currency(self, currency: str) -> str:
        """Get country name from currency code."""
        currency_to_country = {
            'USD': 'United States',
            'EUR': 'Eurozone',
            'GBP': 'United Kingdom',
            'JPY': 'Japan',
            'CHF': 'Switzerland',
            'CAD': 'Canada',
            'AUD': 'Australia',
            'NZD': 'New Zealand'
        }
        return currency_to_country.get(currency, currency)
    
    async def _fetch_economic_indicators(self, currency: str) -> List[EconomicIndicator]:
        """Fetch latest economic indicators for a currency."""
        indicators = []
        
        try:
            if currency in self.economic_indicators:
                for indicator_name, config in self.economic_indicators[currency].items():
                    indicator_data = await self._fetch_single_indicator(
                        currency, indicator_name, config
                    )
                    if indicator_data:
                        indicators.append(indicator_data)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching indicators for {currency}: {str(e)}")
            return []
    
    async def _fetch_single_indicator(self, currency: str, indicator_name: str, 
                                    config: Dict) -> Optional[EconomicIndicator]:
        """Fetch data for a single economic indicator."""
        try:
            # Try FRED API first for USD indicators
            if currency == 'USD' and self.fred and 'fred_id' in config:
                return await self._fetch_from_fred(currency, indicator_name, config)
            
            # Try Alpha Vantage for other indicators
            elif self.alpha_vantage_key and 'symbol' in config:
                return await self._fetch_from_alpha_vantage(currency, indicator_name, config)
            
            # Fallback to mock data for demonstration
            else:
                return self._create_mock_indicator(currency, indicator_name, config)
                
        except Exception as e:
            logger.error(f"Error fetching {indicator_name} for {currency}: {str(e)}")
            return None
    
    async def _fetch_from_fred(self, currency: str, indicator_name: str, 
                             config: Dict) -> Optional[EconomicIndicator]:
        """Fetch indicator data from FRED API."""
        try:
            fred_id = config['fred_id']
            
            # Get latest data
            data = self.fred.get_series(fred_id, limit=12)  # Last 12 observations
            
            if len(data) < 2:
                return None
            
            current_value = float(data.iloc[-1])
            previous_value = float(data.iloc[-2])
            
            # Calculate forecast (simple trend)
            if len(data) >= 3:
                trend = (current_value - data.iloc[-3]) / 2
                forecast = current_value + trend
            else:
                forecast = current_value
            
            # Calculate surprise factor
            surprise_factor = abs(current_value - forecast) / max(abs(forecast), 0.1)
            
            # Determine impact
            impact = self._calculate_indicator_impact(
                current_value, previous_value, forecast, indicator_name
            )
            
            # Get historical correlation
            correlation = self.indicator_correlations.get(currency, {}).get(indicator_name, 0.5)
            
            return EconomicIndicator(
                name=indicator_name,
                country=self._get_country_from_currency(currency),
                value=current_value,
                previous_value=previous_value,
                forecast=forecast,
                release_date=data.index[-1],
                importance=config['importance'],
                impact=impact,
                surprise_factor=surprise_factor,
                historical_correlation=correlation,
                market_reaction_potential=surprise_factor * correlation
            )
            
        except Exception as e:
            logger.error(f"Error fetching from FRED: {str(e)}")
            return None
    
    async def _fetch_from_alpha_vantage(self, currency: str, indicator_name: str, 
                                      config: Dict) -> Optional[EconomicIndicator]:
        """Fetch indicator data from Alpha Vantage API."""
        try:
            # Alpha Vantage implementation would go here
            # For now, return mock data
            return self._create_mock_indicator(currency, indicator_name, config)
            
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
            return None
    
    def _create_mock_indicator(self, currency: str, indicator_name: str, 
                             config: Dict) -> EconomicIndicator:
        """Create mock indicator data for demonstration."""
        # Generate realistic mock data based on indicator type
        base_values = {
            'gdp': 2.5, 'unemployment': 4.0, 'inflation': 3.2, 'fed_rate': 5.25,
            'nonfarm_payrolls': 150000, 'trade_balance': -80000,
            'retail_sales': 0.5, 'manufacturing_pmi': 52.0
        }
        
        base_value = base_values.get(indicator_name.split('_')[0], 50.0)
        
        # Add some randomness
        current_value = base_value * (1 + np.random.uniform(-0.1, 0.1))
        previous_value = base_value * (1 + np.random.uniform(-0.1, 0.1))
        forecast = base_value * (1 + np.random.uniform(-0.05, 0.05))
        
        surprise_factor = abs(current_value - forecast) / max(abs(forecast), 0.1)
        
        impact = self._calculate_indicator_impact(
            current_value, previous_value, forecast, indicator_name
        )
        
        correlation = self.indicator_correlations.get(currency, {}).get(indicator_name, 0.5)
        
        return EconomicIndicator(
            name=indicator_name,
            country=self._get_country_from_currency(currency),
            value=current_value,
            previous_value=previous_value,
            forecast=forecast,
            release_date=datetime.now() - timedelta(days=np.random.randint(1, 30)),
            importance=config['importance'],
            impact=impact,
            surprise_factor=surprise_factor,
            historical_correlation=correlation,
            market_reaction_potential=surprise_factor * correlation
        )
    
    def _calculate_indicator_impact(self, current: float, previous: float, 
                                  forecast: float, indicator_name: str) -> EconomicImpact:
        """Calculate the economic impact of an indicator value."""
        try:
            # For most indicators, higher is better
            positive_indicators = ['gdp', 'nonfarm_payrolls', 'retail_sales', 'manufacturing_pmi']
            # For these indicators, lower is better
            negative_indicators = ['unemployment', 'inflation']
            # For interest rates, the impact depends on context
            rate_indicators = ['fed_rate', 'ecb_rate', 'boe_rate', 'boj_rate']
            
            # Calculate change from previous and deviation from forecast
            change_from_previous = (current - previous) / max(abs(previous), 0.1)
            deviation_from_forecast = (current - forecast) / max(abs(forecast), 0.1)
            
            # Combined impact score
            if any(ind in indicator_name for ind in positive_indicators):
                impact_score = (change_from_previous * 0.6) + (deviation_from_forecast * 0.4)
            elif any(ind in indicator_name for ind in negative_indicators):
                impact_score = -(change_from_previous * 0.6) - (deviation_from_forecast * 0.4)
            elif any(ind in indicator_name for ind in rate_indicators):
                # For rates, positive change is generally positive for currency
                impact_score = (change_from_previous * 0.7) + (deviation_from_forecast * 0.3)
            else:
                # Default case
                impact_score = deviation_from_forecast
            
            # Convert to EconomicImpact
            if impact_score >= 0.15:
                return EconomicImpact.VERY_POSITIVE
            elif impact_score >= 0.08:
                return EconomicImpact.POSITIVE
            elif impact_score >= 0.03:
                return EconomicImpact.SLIGHTLY_POSITIVE
            elif impact_score >= -0.03:
                return EconomicImpact.NEUTRAL
            elif impact_score >= -0.08:
                return EconomicImpact.SLIGHTLY_NEGATIVE
            elif impact_score >= -0.15:
                return EconomicImpact.NEGATIVE
            else:
                return EconomicImpact.VERY_NEGATIVE
                
        except Exception as e:
            logger.error(f"Error calculating indicator impact: {str(e)}")
            return EconomicImpact.NEUTRAL
    
    async def _analyze_central_bank_policy(self, currency: str) -> CentralBankPolicy:
        """Analyze central bank policy for the currency."""
        try:
            cb_info = self.central_banks.get(currency, {})
            
            # Get current interest rate
            current_rate = await self._get_current_interest_rate(currency)
            previous_rate = await self._get_previous_interest_rate(currency)
            
            # Determine policy stance
            policy_stance = self._determine_policy_stance(current_rate, previous_rate, currency)
            
            # Calculate next meeting date
            next_meeting = self._calculate_next_meeting_date(currency)
            
            # Estimate rate change probabilities
            rate_probabilities = self._calculate_rate_change_probabilities(
                currency, current_rate, policy_stance
            )
            
            # Get forward guidance (mock for now)
            forward_guidance = self._get_forward_guidance(currency, policy_stance)
            
            # Assess overall impact
            impact = self._assess_cb_policy_impact(policy_stance, rate_probabilities)
            
            return CentralBankPolicy(
                bank=cb_info.get('name', f'{currency} Central Bank'),
                country=self._get_country_from_currency(currency),
                current_rate=current_rate,
                previous_rate=previous_rate,
                policy_stance=policy_stance,
                next_meeting=next_meeting,
                rate_change_probability=rate_probabilities,
                forward_guidance=forward_guidance,
                impact_assessment=impact
            )
            
        except Exception as e:
            logger.error(f"Error analyzing central bank policy for {currency}: {str(e)}")
            return self._create_neutral_cb_policy(currency)
    
    async def _get_current_interest_rate(self, currency: str) -> float:
        """Get current interest rate for currency."""
        # Mock rates for demonstration
        rates = {
            'USD': 5.25, 'EUR': 4.50, 'GBP': 5.25, 'JPY': -0.10,
            'CHF': 1.75, 'CAD': 5.00, 'AUD': 4.35, 'NZD': 5.50
        }
        return rates.get(currency, 2.5)
    
    async def _get_previous_interest_rate(self, currency: str) -> float:
        """Get previous interest rate for currency."""
        current = await self._get_current_interest_rate(currency)
        # Mock: assume previous rate was 0.25% different
        return current - 0.25
    
    def _determine_policy_stance(self, current_rate: float, previous_rate: float, 
                               currency: str) -> str:
        """Determine central bank policy stance."""
        rate_change = current_rate - previous_rate
        
        if rate_change > 0.1:
            return 'hawkish'
        elif rate_change < -0.1:
            return 'dovish'
        else:
            # Look at absolute rate level
            if current_rate > 4.0:
                return 'hawkish'
            elif current_rate < 1.0:
                return 'dovish'
            else:
                return 'neutral'
    
    def _calculate_next_meeting_date(self, currency: str) -> datetime:
        """Calculate next central bank meeting date."""
        cb_info = self.central_banks.get(currency, {})
        meeting_months = cb_info.get('typical_meeting_months', [])
        
        if not meeting_months:
            # Default: assume monthly meetings
            next_month = datetime.now().replace(day=1) + timedelta(days=32)
            return next_month.replace(day=15)  # Mid-month
        
        # Find next meeting month
        current_month = datetime.now().month
        next_meeting_month = None
        
        for month in meeting_months:
            if month > current_month:
                next_meeting_month = month
                break
        
        if next_meeting_month is None:
            # Next year
            next_meeting_month = meeting_months[0]
            year = datetime.now().year + 1
        else:
            year = datetime.now().year
        
        return datetime(year, next_meeting_month, 15)  # Assume mid-month
    
    def _calculate_rate_change_probabilities(self, currency: str, current_rate: float, 
                                           stance: str) -> Dict[str, float]:
        """Calculate probabilities of rate changes."""
        if stance == 'hawkish':
            return {'hike': 0.7, 'hold': 0.25, 'cut': 0.05}
        elif stance == 'dovish':
            return {'hike': 0.05, 'hold': 0.25, 'cut': 0.7}
        else:
            return {'hike': 0.2, 'hold': 0.6, 'cut': 0.2}
    
    def _get_forward_guidance(self, currency: str, stance: str) -> str:
        """Get forward guidance statement."""
        if stance == 'hawkish':
            return "Policy remains focused on bringing inflation down to target through restrictive monetary policy."
        elif stance == 'dovish':
            return "Policy will remain accommodative to support economic growth and employment."
        else:
            return "Policy stance will be data-dependent and responsive to economic conditions."
    
    def _assess_cb_policy_impact(self, stance: str, probabilities: Dict[str, float]) -> EconomicImpact:
        """Assess overall impact of central bank policy."""
        if stance == 'hawkish' and probabilities.get('hike', 0) > 0.6:
            return EconomicImpact.POSITIVE
        elif stance == 'dovish' and probabilities.get('cut', 0) > 0.6:
            return EconomicImpact.NEGATIVE
        else:
            return EconomicImpact.NEUTRAL
    
    def _create_neutral_cb_policy(self, currency: str) -> CentralBankPolicy:
        """Create neutral central bank policy for fallback."""
        return CentralBankPolicy(
            bank=f'{currency} Central Bank',
            country=self._get_country_from_currency(currency),
            current_rate=2.5,
            previous_rate=2.5,
            policy_stance='neutral',
            next_meeting=datetime.now() + timedelta(days=45),
            rate_change_probability={'hike': 0.3, 'hold': 0.4, 'cut': 0.3},
            forward_guidance="Policy will be data-dependent.",
            impact_assessment=EconomicImpact.NEUTRAL
        )
    
    def _calculate_economic_health(self, indicators: List[EconomicIndicator], 
                                 cb_policy: CentralBankPolicy) -> EconomicImpact:
        """Calculate overall economic health score."""
        if not indicators:
            return EconomicImpact.NEUTRAL
        
        # Weight indicators by importance
        importance_weights = {
            IndicatorImportance.CRITICAL: 1.0,
            IndicatorImportance.HIGH: 0.7,
            IndicatorImportance.MEDIUM: 0.4,
            IndicatorImportance.LOW: 0.2
        }
        
        # Convert impacts to scores
        impact_scores = {
            EconomicImpact.VERY_POSITIVE: 1.0,
            EconomicImpact.POSITIVE: 0.6,
            EconomicImpact.SLIGHTLY_POSITIVE: 0.3,
            EconomicImpact.NEUTRAL: 0.0,
            EconomicImpact.SLIGHTLY_NEGATIVE: -0.3,
            EconomicImpact.NEGATIVE: -0.6,
            EconomicImpact.VERY_NEGATIVE: -1.0
        }
        
        total_score = 0
        total_weight = 0
        
        for indicator in indicators:
            weight = importance_weights.get(indicator.importance, 0.5)
            score = impact_scores.get(indicator.impact, 0.0)
            total_score += score * weight
            total_weight += weight
        
        # Add central bank policy impact
        cb_score = impact_scores.get(cb_policy.impact_assessment, 0.0)
        total_score += cb_score * 0.5  # CB policy gets significant weight
        total_weight += 0.5
        
        if total_weight == 0:
            return EconomicImpact.NEUTRAL
        
        average_score = total_score / total_weight
        
        # Convert back to impact
        if average_score >= 0.6:
            return EconomicImpact.VERY_POSITIVE
        elif average_score >= 0.3:
            return EconomicImpact.POSITIVE
        elif average_score >= 0.1:
            return EconomicImpact.SLIGHTLY_POSITIVE
        elif average_score >= -0.1:
            return EconomicImpact.NEUTRAL
        elif average_score >= -0.3:
            return EconomicImpact.SLIGHTLY_NEGATIVE
        elif average_score >= -0.6:
            return EconomicImpact.NEGATIVE
        else:
            return EconomicImpact.VERY_NEGATIVE
    
    def _analyze_gdp_trend(self, indicators: List[EconomicIndicator]) -> str:
        """Analyze GDP growth trend."""
        gdp_indicators = [ind for ind in indicators if 'gdp' in ind.name.lower()]
        
        if not gdp_indicators:
            return "unknown"
        
        gdp = gdp_indicators[0]
        change = ((gdp.value - gdp.previous_value) / max(abs(gdp.previous_value), 0.1)) * 100
        
        if change > 0.5:
            return "accelerating"
        elif change > 0:
            return "growing"
        elif change > -0.5:
            return "slowing"
        else:
            return "contracting"
    
    def _analyze_inflation_trend(self, indicators: List[EconomicIndicator]) -> str:
        """Analyze inflation trend."""
        inflation_indicators = [ind for ind in indicators if 'inflation' in ind.name.lower()]
        
        if not inflation_indicators:
            return "unknown"
        
        inflation = inflation_indicators[0]
        
        if inflation.value > 4.0:
            return "high"
        elif inflation.value > 2.5:
            return "above_target"
        elif inflation.value > 1.5:
            return "target_range"
        else:
            return "low"
    
    def _analyze_employment_trend(self, indicators: List[EconomicIndicator]) -> str:
        """Analyze employment trend."""
        employment_indicators = [ind for ind in indicators 
                               if 'unemployment' in ind.name.lower() or 'payroll' in ind.name.lower()]
        
        if not employment_indicators:
            return "unknown"
        
        # For unemployment, lower is better
        for indicator in employment_indicators:
            if 'unemployment' in indicator.name.lower():
                change = indicator.value - indicator.previous_value
                if change < -0.2:
                    return "improving"
                elif change < 0:
                    return "stable_improving"
                elif change < 0.2:
                    return "stable"
                else:
                    return "deteriorating"
        
        # For payrolls, higher is better
        for indicator in employment_indicators:
            if 'payroll' in indicator.name.lower():
                change = ((indicator.value - indicator.previous_value) / 
                         max(abs(indicator.previous_value), 1000)) * 100
                if change > 2:
                    return "strong"
                elif change > 0:
                    return "improving"
                else:
                    return "weakening"
        
        return "stable"
    
    def _determine_monetary_policy_bias(self, cb_policy: CentralBankPolicy, 
                                      indicators: List[EconomicIndicator]) -> str:
        """Determine overall monetary policy bias."""
        # Start with central bank's stated stance
        base_bias = cb_policy.policy_stance
        
        # Adjust based on economic indicators
        inflation_high = any(ind.value > 3.0 for ind in indicators if 'inflation' in ind.name.lower())
        unemployment_high = any(ind.value > 5.0 for ind in indicators if 'unemployment' in ind.name.lower())
        
        if inflation_high and not unemployment_high:
            if base_bias == 'neutral':
                return 'hawkish'
            elif base_bias == 'dovish':
                return 'neutral'
        
        elif unemployment_high and not inflation_high:
            if base_bias == 'neutral':
                return 'dovish'
            elif base_bias == 'hawkish':
                return 'neutral'
        
        return base_bias
    
    def _assess_fiscal_policy_impact(self, currency: str, 
                                   indicators: List[EconomicIndicator]) -> EconomicImpact:
        """Assess fiscal policy impact (simplified)."""
        # This would typically involve government debt, deficit, spending data
        # For now, return neutral as we don't have detailed fiscal data
        return EconomicImpact.NEUTRAL
    
    def _analyze_trade_balance_impact(self, currency: str, 
                                    indicators: List[EconomicIndicator]) -> EconomicImpact:
        """Analyze trade balance impact."""
        trade_indicators = [ind for ind in indicators if 'trade' in ind.name.lower()]
        
        if not trade_indicators:
            return EconomicImpact.NEUTRAL
        
        trade_balance = trade_indicators[0]
        
        # Positive trade balance is good for currency
        if trade_balance.value > 0:
            return EconomicImpact.POSITIVE if trade_balance.value > trade_balance.previous_value else EconomicImpact.SLIGHTLY_POSITIVE
        else:
            # Negative trade balance - check if improving
            if trade_balance.value > trade_balance.previous_value:
                return EconomicImpact.SLIGHTLY_POSITIVE
            else:
                return EconomicImpact.SLIGHTLY_NEGATIVE
    
    def _identify_risk_factors(self, currency: str, indicators: List[EconomicIndicator], 
                             cb_policy: CentralBankPolicy) -> List[str]:
        """Identify economic risk factors."""
        risks = []
        
        # High inflation risk
        inflation_indicators = [ind for ind in indicators if 'inflation' in ind.name.lower()]
        if inflation_indicators and inflation_indicators[0].value > 4.0:
            risks.append("High inflation above central bank target")
        
        # High unemployment risk
        unemployment_indicators = [ind for ind in indicators if 'unemployment' in ind.name.lower()]
        if unemployment_indicators and unemployment_indicators[0].value > 6.0:
            risks.append("Elevated unemployment levels")
        
        # Policy uncertainty
        if cb_policy.rate_change_probability.get('hike', 0) > 0.6 and cb_policy.rate_change_probability.get('cut', 0) > 0.3:
            risks.append("Central bank policy uncertainty")
        
        # Economic slowdown
        gdp_indicators = [ind for ind in indicators if 'gdp' in ind.name.lower()]
        if gdp_indicators and gdp_indicators[0].value < gdp_indicators[0].previous_value:
            risks.append("Economic growth slowdown")
        
        return risks
    
    def _identify_opportunities(self, currency: str, indicators: List[EconomicIndicator], 
                              cb_policy: CentralBankPolicy) -> List[str]:
        """Identify economic opportunities."""
        opportunities = []
        
        # Strong employment
        unemployment_indicators = [ind for ind in indicators if 'unemployment' in ind.name.lower()]
        if unemployment_indicators and unemployment_indicators[0].value < 4.0:
            opportunities.append("Strong employment market supporting consumption")
        
        # Controlled inflation
        inflation_indicators = [ind for ind in indicators if 'inflation' in ind.name.lower()]
        if inflation_indicators and 1.5 <= inflation_indicators[0].value <= 2.5:
            opportunities.append("Inflation within target range supporting stable policy")
        
        # Clear policy direction
        if cb_policy.rate_change_probability.get('hike', 0) > 0.7:
            opportunities.append("Clear hawkish policy stance supporting currency")
        elif cb_policy.rate_change_probability.get('cut', 0) > 0.7:
            opportunities.append("Accommodative policy supporting economic growth")
        
        # GDP growth
        gdp_indicators = [ind for ind in indicators if 'gdp' in ind.name.lower()]
        if gdp_indicators and gdp_indicators[0].value > gdp_indicators[0].previous_value:
            opportunities.append("Positive economic growth momentum")
        
        return opportunities
    
    def _generate_currency_outlook(self, currency: str, indicators: List[EconomicIndicator], 
                                 cb_policy: CentralBankPolicy, health: EconomicImpact) -> Dict[str, float]:
        """Generate currency outlook for different timeframes."""
        # Base outlook from economic health
        health_scores = {
            EconomicImpact.VERY_POSITIVE: 0.8,
            EconomicImpact.POSITIVE: 0.5,
            EconomicImpact.SLIGHTLY_POSITIVE: 0.2,
            EconomicImpact.NEUTRAL: 0.0,
            EconomicImpact.SLIGHTLY_NEGATIVE: -0.2,
            EconomicImpact.NEGATIVE: -0.5,
            EconomicImpact.VERY_NEGATIVE: -0.8
        }
        
        base_score = health_scores.get(health, 0.0)
        
        # Adjust for monetary policy
        policy_adjustment = 0
        if cb_policy.policy_stance == 'hawkish':
            policy_adjustment = 0.3
        elif cb_policy.policy_stance == 'dovish':
            policy_adjustment = -0.3
        
        # Generate outlook for different timeframes
        outlook = {
            '1m': base_score + policy_adjustment,
            '3m': base_score + (policy_adjustment * 0.7),
            '6m': base_score + (policy_adjustment * 0.4)
        }
        
        # Bound between -1 and 1
        for timeframe in outlook:
            outlook[timeframe] = max(-1.0, min(1.0, outlook[timeframe]))
        
        return outlook
    
    def _create_neutral_analysis(self, currency: str) -> EconomicAnalysis:
        """Create neutral economic analysis for fallback."""
        return EconomicAnalysis(
            country=self._get_country_from_currency(currency),
            currency=currency,
            overall_health=EconomicImpact.NEUTRAL,
            gdp_growth_trend="unknown",
            inflation_trend="unknown",
            employment_trend="unknown",
            monetary_policy_bias="neutral",
            fiscal_policy_impact=EconomicImpact.NEUTRAL,
            trade_balance_impact=EconomicImpact.NEUTRAL,
            key_indicators=[],
            central_bank_policy=self._create_neutral_cb_policy(currency),
            risk_factors=["Limited economic data available"],
            opportunities=[],
            currency_outlook={'1m': 0.0, '3m': 0.0, '6m': 0.0}
        )
    
    async def _send_economic_signal(self, currency: str, analysis: EconomicAnalysis):
        """Send economic analysis signal to other agents."""
        try:
            signal_data = {
                'currency': currency,
                'overall_health': analysis.overall_health.value,
                'monetary_policy_bias': analysis.monetary_policy_bias,
                'currency_outlook': analysis.currency_outlook,
                'key_risks': analysis.risk_factors[:3],  # Top 3 risks
                'opportunities': analysis.opportunities[:3],  # Top 3 opportunities
                'central_bank_stance': analysis.central_bank_policy.policy_stance,
                'next_cb_meeting': analysis.central_bank_policy.next_meeting.isoformat(),
                'source': 'economic_factors'
            }
            
            # Calculate confidence based on data quality
            confidence = len(analysis.key_indicators) / 5.0  # Assume 5 is ideal number of indicators
            confidence = min(max(confidence, 0.3), 1.0)  # Bound between 0.3 and 1.0
            
            message = {
                'sender': self.agent_id,
                'receiver': 'risk_manager',
                'message_type': 'economic_analysis',
                'data': signal_data,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to risk manager
            channel = "agent_messages_risk_manager"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            # Send to trend analyzer for confluence
            message['receiver'] = 'trend_identifier'
            channel = "agent_messages_trend_identifier"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            logger.info(f"Economic signal sent: {analysis.overall_health.value} for {currency}")
            
        except Exception as e:
            logger.error(f"Error sending economic signal: {str(e)}")
    
    def update_performance(self, prediction_result: Dict):
        """Update performance metrics based on actual results."""
        try:
            self.performance_metrics['total_analyses'] += 1
            
            if prediction_result.get('accurate', False):
                self.performance_metrics['accurate_predictions'] += 1
            
            self.performance_metrics['accuracy'] = (
                self.performance_metrics['accurate_predictions'] / 
                self.performance_metrics['total_analyses']
            )
            
            logger.info(f"Economic analysis performance updated: {self.performance_metrics['accuracy']:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """Get comprehensive economic analysis summary."""
        return {
            'agent_id': self.agent_id,
            'performance': self.performance_metrics,
            'total_analyses': len(self.analysis_history),
            'recent_analyses': [
                {
                    'currency': analysis['currency'],
                    'health': analysis['analysis'].overall_health.value,
                    'outlook_1m': analysis['analysis'].currency_outlook.get('1m', 0.0),
                    'timestamp': analysis['timestamp']
                }
                for analysis in self.analysis_history[-5:]
            ],
            'supported_currencies': list(self.economic_indicators.keys())
        }