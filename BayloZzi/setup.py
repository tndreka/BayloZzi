"""
Forex Multi-Agent Trading System
Setup configuration for easy installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="forex-multi-agent-trading",
    version="1.0.0",
    author="BayloZzi Trading Systems",
    author_email="contact@baylozzi.com",
    description="A sophisticated forex trading system using multiple AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baylozzi/forex-trading",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.5",
            "pytest-asyncio>=0.24.0",
            "black>=24.10.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "docker": [
            "docker>=7.0.0",
            "docker-compose>=1.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forex-backtest=run.backtest:run_backtest",
            "forex-live=run.live_trade:main",
            "forex-server=BayloZzi.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.pkl", "*.csv", "*.json", "*.yml", "*.yaml"],
        "models": ["*.pkl"],
        "data": ["*.csv"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/baylozzi/forex-trading/issues",
        "Source": "https://github.com/baylozzi/forex-trading",
        "Documentation": "https://docs.baylozzi.com",
    },
)