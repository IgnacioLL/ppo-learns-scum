from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scum_agent",  # Replace with your package name
    version="0.1.0",  # Initial version number
    author="Ignacio Lloret Lorente",  # Replace with your name
    author_email="ig.lloret.l@gmail.com",  # Replace with your email
    description="A2C agent for reinforcement learning to play Scum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IgnacioLL/ppo-agent-plays-scum",  # Replace with your repository URL
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),  # Automatically find and include your packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Assuming MIT, adjust if different
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',  # Specify the Python versions that are compatible
)
