#!/bin/bash
# Generate secure secrets for Langfuse Docker Compose setup

echo "# Add these to your .env file:"
echo ""
echo "NEXTAUTH_SECRET=$(openssl rand -base64 32)"
echo "SALT=$(openssl rand -base64 32)"
echo "ENCRYPTION_KEY=$(openssl rand -hex 32)"
echo "NEXTAUTH_URL=http://localhost:3000"

