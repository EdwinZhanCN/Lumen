#!/bin/bash
# Lumen-CLIP Service Startup Script
#
# This script provides a convenient way to start Lumen-CLIP services with
# proper validation, environment setup, and error handling.
#
# Usage:
#   ./scripts/start_service.sh --config config/clip_only.yaml
#   ./scripts/start_service.sh --config config/bioclip_only.yaml --port 50052
#   ./scripts/start_service.sh --config config/unified_service.yaml --validate-only

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
CONFIG_FILE=""
PORT=""
VALIDATE_ONLY=false
LOG_LEVEL="INFO"
SKIP_VALIDATION=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display usage
usage() {
    cat << EOF
Lumen-CLIP Service Startup Script

Usage:
    $0 --config <path> [OPTIONS]

Required:
    --config <path>         Path to YAML configuration file

Optional:
    --port <number>         Override port from config file
    --log-level <level>     Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                            Default: INFO
    --validate-only         Only validate config without starting service
    --skip-validation       Skip config validation (not recommended)
    --advertise-ip <ip>     Override advertised IP for mDNS
    --help                  Show this help message

Environment Variables:
    ADVERTISE_IP            Override advertised IP for mDNS
    SERVICE_UUID            Set custom service UUID
    SERVICE_VERSION         Set custom service version
    PYTHONPATH              Python path (auto-set to project src/)

Examples:
    # Start CLIP service with default settings
    $0 --config config/clip_only.yaml

    # Start BioCLIP service on custom port
    $0 --config config/bioclip_only.yaml --port 50052

    # Validate config without starting
    $0 --config config/unified_service.yaml --validate-only

    # Start with custom advertised IP for mDNS
    $0 --config config/clip_only.yaml --advertise-ip 192.168.1.100

    # Enable debug logging
    $0 --config config/clip_only.yaml --log-level DEBUG

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --advertise-ip)
            export ADVERTISE_IP="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG_FILE" ]; then
    print_error "Configuration file is required"
    usage
fi

# Check if config file exists
if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Resolve config file path
if [ -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi

print_info "Lumen-CLIP Service Startup"
echo ""

# Step 1: Check Python
print_info "Checking Python environment..."
if ! command_exists python3; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

# Step 2: Check if in virtual environment (recommended)
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Not running in a virtual environment"
    print_warning "Consider activating a venv: source .venv/bin/activate"
fi

# Step 3: Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
print_success "PYTHONPATH set to: $PROJECT_ROOT/src"

# Step 4: Check required dependencies
print_info "Checking dependencies..."
MISSING_DEPS=()

if ! python3 -c "import grpc" 2>/dev/null; then
    MISSING_DEPS+=("grpcio")
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    MISSING_DEPS+=("pyyaml")
fi

if ! python3 -c "import torch" 2>/dev/null; then
    MISSING_DEPS+=("torch")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_error "Missing dependencies: ${MISSING_DEPS[*]}"
    print_info "Install with: pip install ${MISSING_DEPS[*]}"
    exit 1
fi

print_success "All core dependencies found"

# Step 5: Check for lumen-resources
print_info "Checking lumen-resources availability..."
if command_exists lumen-resources; then
    print_success "lumen-resources command found"
elif python3 -c "import lumen_resources" 2>/dev/null; then
    print_success "lumen-resources module found"
    # Create alias for lumen-resources command
    lumen-resources() {
        python3 -m lumen_resources "$@"
    }
else
    print_warning "lumen-resources not found"
    print_warning "Some validation features may not be available"
fi

# Step 6: Validate configuration
if [ "$SKIP_VALIDATION" = false ]; then
    print_info "Validating configuration: $CONFIG_FILE"

    if command_exists lumen-resources; then
        if lumen-resources validate "$CONFIG_FILE"; then
            print_success "Configuration validation passed"
        else
            print_error "Configuration validation failed"
            exit 1
        fi
    else
        print_warning "Skipping detailed validation (lumen-resources not available)"
        # Basic YAML syntax check
        if python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" 2>/dev/null; then
            print_success "Configuration YAML syntax is valid"
        else
            print_error "Configuration YAML syntax is invalid"
            exit 1
        fi
    fi
else
    print_warning "Skipping configuration validation"
fi

# Step 7: Exit if validate-only mode
if [ "$VALIDATE_ONLY" = true ]; then
    print_success "Validation complete. Exiting (--validate-only mode)"
    exit 0
fi

# Step 8: Display configuration summary
print_info "Configuration summary:"
echo "  Config file: $CONFIG_FILE"
echo "  Log level: $LOG_LEVEL"
if [ -n "$PORT" ]; then
    echo "  Port override: $PORT"
fi
if [ -n "$ADVERTISE_IP" ]; then
    echo "  Advertised IP: $ADVERTISE_IP"
fi
echo ""

# Step 9: Start the service
print_info "Starting Lumen-CLIP service..."
echo ""

# Build command
CMD="python3 $PROJECT_ROOT/src/server.py --config $CONFIG_FILE --log-level $LOG_LEVEL"

if [ -n "$PORT" ]; then
    CMD="$CMD --port $PORT"
fi

# Display command
print_info "Executing: $CMD"
echo ""

# Set up signal handling for graceful shutdown
trap 'print_warning "Shutting down service..."; exit 0' SIGINT SIGTERM

# Execute
exec $CMD
