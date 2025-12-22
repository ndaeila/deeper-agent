# Langfuse Setup Guide

Langfuse provides observability for your LLM application, allowing you to trace every call, see token usage, latency, and debug issues.

## Quick Start

### 1. Start Langfuse with Docker Compose

```bash
docker compose up -d
```

This starts:
- **Langfuse Server** at http://localhost:3000
- **PostgreSQL** database for persistence

### 2. Create Your Account & Project

1. Open http://localhost:3000
2. Sign up for an account (local, no cloud)
3. Create a new project
4. Go to **Settings → API Keys** and create new keys

### 3. Configure Environment Variables

Create a `.env` file in your project or `examples/test_retriever/` directory:

```bash
# Model Configuration
MODEL_NAME=gpt-4
MODEL_URL=https://api.openai.com/v1
API_KEY=sk-your-api-key-here

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_HOST=http://localhost:3000
```

### 4. Run Your Agent

```bash
poetry run python examples/test_retriever/test_retriever.py --query "Your question"
```

You'll see traces appear in the Langfuse dashboard at http://localhost:3000.

## What You Can See in Langfuse

- **Traces**: Complete execution flow of your agent
- **Spans**: Individual LLM calls, tool executions
- **Token Usage**: Input/output tokens per call
- **Latency**: Time taken for each operation
- **Costs**: Estimated costs per model call
- **Scores**: Add custom evaluations

## Docker Compose Configuration

The `docker-compose.yml` at the project root includes:

| Service | Port | Description |
|---------|------|-------------|
| `langfuse-server` | 3000 | Langfuse web UI and API |
| `db` | 5432 | PostgreSQL database |

### Persistent Data

PostgreSQL data is stored in a Docker volume (`langfuse_postgres_data`). To reset:

```bash
docker compose down -v  # Removes volumes
docker compose up -d    # Fresh start
```

## Disabling Langfuse

If you want to run without tracing:

```bash
poetry run python examples/test_retriever/test_retriever.py --no-langfuse
```

Or simply don't set the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` environment variables.

## Production Deployment

For production, consider:
- Using Langfuse Cloud (https://langfuse.com)
- Or self-hosting with proper secrets and HTTPS

Update these environment variables for production:

```bash
NEXTAUTH_SECRET=<generate-a-secure-random-string>
SALT=<generate-a-secure-random-string>
ENCRYPTION_KEY=<64-character-hex-string>
```

## Troubleshooting

### "Langfuse not installed"
```bash
poetry add langfuse
```

### Can't connect to Langfuse
1. Check Docker is running: `docker compose ps`
2. Check logs: `docker compose logs langfuse-server`
3. Ensure port 3000 is not in use

### Traces not appearing
1. Ensure you call `flush()` before exiting (done automatically in test script)
2. Check your API keys are correct
3. Verify `LANGFUSE_HOST` matches your setup

### JWT Session Error (decryption operation failed)
This error occurs when NextAuth cannot decrypt session tokens. Common causes:

1. **Missing or changed secrets**: Ensure your `.env` file has valid values for:
   - `NEXTAUTH_SECRET` (32+ character random string)
   - `SALT` (32+ character random string)
   - `ENCRYPTION_KEY` (64-character hex string)
   - `NEXTAUTH_URL` (should be `http://localhost:3000`)

2. **Stale browser session**: Clear cookies for `localhost:3000` or use incognito mode

3. **Generate new secrets**: Run the helper script:
   ```bash
   ./scripts/generate-langfuse-secrets.sh
   ```
   Then add the output to your `.env` file and restart:
   ```bash
   docker compose down
   docker compose up -d
   ```

**Note**: Changing secrets will invalidate existing sessions. Users will need to log in again.

