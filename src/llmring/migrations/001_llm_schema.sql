-- LLM Service Database Schema (schema-aware)
-- Models registry and API call tracking

-- Enable necessary extensions (in public schema)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =====================================================
-- UTILITY FUNCTIONS
-- =====================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION {{schema}}.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- LLM MODELS REGISTRY
-- =====================================================

-- Available LLM models with cost and capability information
CREATE TABLE IF NOT EXISTS {{tables.llm_models}} (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL CHECK (provider IN ('anthropic', 'openai', 'google', 'ollama')),
    model_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(255),
    description TEXT,

    -- Model capabilities
    max_context INTEGER, -- renamed from context_length for compatibility
    max_output_tokens INTEGER,
    supports_vision BOOLEAN DEFAULT FALSE,
    supports_function_calling BOOLEAN DEFAULT FALSE,
    supports_json_mode BOOLEAN DEFAULT FALSE,
    supports_parallel_tool_calls BOOLEAN DEFAULT FALSE,
    tool_call_format VARCHAR(50),

    -- Cost information (dollars per million tokens)
    dollars_per_million_tokens_input NUMERIC(12, 6), -- Cost in dollars per 1M input tokens
    dollars_per_million_tokens_output NUMERIC(12, 6), -- Cost in dollars per 1M output tokens

    -- Status
    inactive_from TIMESTAMP WITH TIME ZONE DEFAULT NULL, -- NULL means active, timestamp means inactive since that date
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT llm_models_unique UNIQUE (provider, model_name)
);

-- =====================================================
-- API CALL TRACKING
-- =====================================================

-- Track all LLM API calls with cost and usage information
CREATE TABLE IF NOT EXISTS {{tables.llm_api_calls}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Origin tracking (required fields)
    origin VARCHAR(255) NOT NULL, -- name of the calling program/app
    id_at_origin VARCHAR(255) NOT NULL, -- user identifier at the origin (username, user_id, etc.)

    -- Model and provider information
    model_id INTEGER REFERENCES {{tables.llm_models}}(id),
    provider VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,

    -- Request details
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,

    -- Cost calculation
    estimated_cost DECIMAL(12, 8) DEFAULT 0, -- higher precision for micro-costs
    dollars_per_million_tokens_input_used NUMERIC(12, 6), -- cost rate used for this call
    dollars_per_million_tokens_output_used NUMERIC(12, 6), -- cost rate used for this call

    -- Performance metrics
    response_time_ms INTEGER,

    -- Request metadata
    temperature FLOAT,
    max_tokens INTEGER,
    top_p FLOAT,
    stream BOOLEAN DEFAULT FALSE,
    stop_sequences JSONB,
    system_prompt TEXT,
    system_prompt_hash VARCHAR(64), -- SHA-256 hash of system prompt for privacy
    tools_used JSONB, -- array of tool names used
    json_mode BOOLEAN DEFAULT FALSE,
    response_format JSONB,
    seed INTEGER,
    tool_choice VARCHAR(50),
    parallel_tool_calls BOOLEAN,

    -- Status and error tracking
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'error', 'timeout', 'rate_limited')),
    error_type VARCHAR(100),
    error_message TEXT,

    -- Timestamps
    called_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexes will be created below
    CONSTRAINT llm_api_calls_tokens_check CHECK (total_tokens = prompt_tokens + completion_tokens)
);

-- =====================================================
-- USAGE ANALYTICS
-- =====================================================

-- Daily aggregated usage by origin and user
CREATE TABLE IF NOT EXISTS {{tables.usage_analytics_daily}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Origin and user
    origin VARCHAR(255) NOT NULL,
    id_at_origin VARCHAR(255) NOT NULL,
    date DATE NOT NULL,

    -- Model usage
    provider VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,

    -- Aggregated metrics
    total_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(12, 8) DEFAULT 0,

    -- Performance metrics
    avg_response_time_ms INTEGER,
    success_rate DECIMAL(5, 4), -- percentage as decimal (0.9500 = 95%)

    -- Error summary
    error_count INTEGER DEFAULT 0,
    timeout_count INTEGER DEFAULT 0,
    rate_limit_count INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT usage_analytics_daily_unique UNIQUE (origin, id_at_origin, date, provider, model_name)
);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON {{tables.llm_models}}(provider);
CREATE INDEX IF NOT EXISTS idx_llm_models_active ON {{tables.llm_models}}(inactive_from) WHERE inactive_from IS NULL;
CREATE INDEX IF NOT EXISTS idx_llm_models_provider_active ON {{tables.llm_models}}(provider, inactive_from) WHERE inactive_from IS NULL;

CREATE INDEX IF NOT EXISTS idx_llm_api_calls_origin ON {{tables.llm_api_calls}}(origin);
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_origin_user ON {{tables.llm_api_calls}}(origin, id_at_origin);
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_called_at ON {{tables.llm_api_calls}}(called_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_origin_called_at ON {{tables.llm_api_calls}}(origin, called_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_model ON {{tables.llm_api_calls}}(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_provider_model ON {{tables.llm_api_calls}}(provider, model_name);
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_status ON {{tables.llm_api_calls}}(status);
-- Date index for daily aggregation queries
CREATE INDEX IF NOT EXISTS idx_llm_api_calls_called_at_date ON {{tables.llm_api_calls}}(called_at);

CREATE INDEX IF NOT EXISTS idx_usage_analytics_daily_origin ON {{tables.usage_analytics_daily}}(origin);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_daily_origin_user ON {{tables.usage_analytics_daily}}(origin, id_at_origin);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_daily_date ON {{tables.usage_analytics_daily}}(date DESC);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_daily_provider_model ON {{tables.usage_analytics_daily}}(provider, model_name);

-- =====================================================
-- TRIGGERS
-- =====================================================

CREATE TRIGGER llm_models_updated_at_trigger
BEFORE UPDATE ON {{tables.llm_models}}
FOR EACH ROW EXECUTE FUNCTION {{schema}}.update_updated_at();

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Record an API call with automatic cost calculation
CREATE OR REPLACE FUNCTION {{schema}}.record_api_call(
    p_origin VARCHAR(255),
    p_id_at_origin VARCHAR(255),
    p_provider VARCHAR(50),
    p_model_name VARCHAR(100),
    p_prompt_tokens INTEGER,
    p_completion_tokens INTEGER,
    p_response_time_ms INTEGER DEFAULT NULL,
    p_temperature FLOAT DEFAULT NULL,
    p_max_tokens INTEGER DEFAULT NULL,
    p_system_prompt_hash VARCHAR(64) DEFAULT NULL,
    p_tools_used JSONB DEFAULT NULL,
    p_status VARCHAR(20) DEFAULT 'success',
    p_error_type VARCHAR(100) DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_call_id UUID;
    v_model_id INTEGER;
    v_total_tokens INTEGER;
    v_estimated_cost DECIMAL(12, 8);
    v_cost_input NUMERIC(12, 6);
    v_cost_output NUMERIC(12, 6);
BEGIN
    v_total_tokens := p_prompt_tokens + p_completion_tokens;

    -- Get model information and costs
    SELECT id, dollars_per_million_tokens_input, dollars_per_million_tokens_output
    INTO v_model_id, v_cost_input, v_cost_output
    FROM {{tables.llm_models}}
    WHERE provider = p_provider AND model_name = p_model_name AND inactive_from IS NULL
    LIMIT 1;

    -- Calculate estimated cost (costs are per million tokens, so divide by 1,000,000)
    IF v_cost_input IS NOT NULL AND v_cost_output IS NOT NULL THEN
        v_estimated_cost := (p_prompt_tokens * v_cost_input / 1000000.0) + (p_completion_tokens * v_cost_output / 1000000.0);
    ELSE
        v_estimated_cost := 0;
    END IF;

    -- Insert the API call record
    INSERT INTO {{tables.llm_api_calls}} (
        origin, id_at_origin, model_id, provider, model_name,
        prompt_tokens, completion_tokens, total_tokens,
        estimated_cost, dollars_per_million_tokens_input_used, dollars_per_million_tokens_output_used,
        response_time_ms, temperature, max_tokens, system_prompt_hash, tools_used,
        status, error_type, error_message
    ) VALUES (
        p_origin, p_id_at_origin, v_model_id, p_provider, p_model_name,
        p_prompt_tokens, p_completion_tokens, v_total_tokens,
        v_estimated_cost, v_cost_input, v_cost_output,
        p_response_time_ms, p_temperature, p_max_tokens, p_system_prompt_hash, p_tools_used,
        p_status, p_error_type, p_error_message
    ) RETURNING id INTO v_call_id;

    RETURN v_call_id;
END;
$$ LANGUAGE plpgsql;

-- Aggregate daily analytics
CREATE OR REPLACE FUNCTION {{schema}}.aggregate_daily_analytics(p_date DATE DEFAULT CURRENT_DATE - INTERVAL '1 day')
RETURNS VOID AS $$
BEGIN
    -- Insert or update daily aggregates
    INSERT INTO {{tables.usage_analytics_daily}} (
        origin, id_at_origin, date, provider, model_name,
        total_calls, total_tokens, total_prompt_tokens, total_completion_tokens, total_cost,
        avg_response_time_ms, success_rate, error_count, timeout_count, rate_limit_count
    )
    SELECT
        origin,
        id_at_origin,
        called_at::date as date,
        provider,
        model_name,
        COUNT(*) as total_calls,
        SUM(total_tokens) as total_tokens,
        SUM(prompt_tokens) as total_prompt_tokens,
        SUM(completion_tokens) as total_completion_tokens,
        SUM(estimated_cost) as total_cost,
        AVG(response_time_ms)::INTEGER as avg_response_time_ms,
        (COUNT(*) FILTER (WHERE status = 'success')::DECIMAL / COUNT(*)) as success_rate,
        COUNT(*) FILTER (WHERE status = 'error') as error_count,
        COUNT(*) FILTER (WHERE status = 'timeout') as timeout_count,
        COUNT(*) FILTER (WHERE status = 'rate_limited') as rate_limit_count
    FROM {{tables.llm_api_calls}}
    WHERE called_at::date = p_date
    GROUP BY origin, id_at_origin, called_at::date, provider, model_name
    ON CONFLICT (origin, id_at_origin, date, provider, model_name) DO UPDATE SET
        total_calls = EXCLUDED.total_calls,
        total_tokens = EXCLUDED.total_tokens,
        total_prompt_tokens = EXCLUDED.total_prompt_tokens,
        total_completion_tokens = EXCLUDED.total_completion_tokens,
        total_cost = EXCLUDED.total_cost,
        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
        success_rate = EXCLUDED.success_rate,
        error_count = EXCLUDED.error_count,
        timeout_count = EXCLUDED.timeout_count,
        rate_limit_count = EXCLUDED.rate_limit_count;
END;
$$ LANGUAGE plpgsql;

-- Get usage statistics for an origin and user
CREATE OR REPLACE FUNCTION {{schema}}.get_usage_stats(
    p_origin VARCHAR(255),
    p_id_at_origin VARCHAR(255),
    p_days INTEGER DEFAULT 30
) RETURNS TABLE (
    total_calls BIGINT,
    total_tokens BIGINT,
    total_cost DECIMAL(12, 8),
    avg_cost_per_call DECIMAL(12, 8),
    most_used_model VARCHAR(100),
    success_rate DECIMAL(5, 4),
    avg_response_time_ms INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_calls,
        SUM(ac.total_tokens)::BIGINT as total_tokens,
        SUM(ac.estimated_cost) as total_cost,
        CASE
            WHEN COUNT(*) > 0 THEN (SUM(ac.estimated_cost) / COUNT(*))
            ELSE 0::DECIMAL
        END as avg_cost_per_call,
        (
            SELECT model_name
            FROM {{tables.llm_api_calls}}
            WHERE origin = p_origin AND id_at_origin = p_id_at_origin
                AND called_at >= CURRENT_DATE - INTERVAL '1 day' * p_days
            GROUP BY model_name
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ) as most_used_model,
        CASE
            WHEN COUNT(*) > 0 THEN (COUNT(*) FILTER (WHERE ac.status = 'success')::DECIMAL / COUNT(*))
            ELSE 0::DECIMAL
        END as success_rate,
        AVG(ac.response_time_ms)::INTEGER as avg_response_time_ms
    FROM {{tables.llm_api_calls}} ac
    WHERE ac.origin = p_origin
        AND ac.id_at_origin = p_id_at_origin
        AND ac.called_at >= CURRENT_DATE - INTERVAL '1 day' * p_days;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert default LLM models (costs in dollars per million tokens)
INSERT INTO {{tables.llm_models}} (provider, model_name, display_name, description, max_context, max_output_tokens, supports_vision, supports_function_calling, supports_json_mode, supports_parallel_tool_calls, dollars_per_million_tokens_input, dollars_per_million_tokens_output) VALUES
-- Anthropic models
('anthropic', 'claude-3-opus-20240229', 'Claude 3 Opus', 'Most capable Claude 3 model', 200000, 4096, TRUE, TRUE, FALSE, FALSE, 15.00, 75.00),
('anthropic', 'claude-3-sonnet-20240229', 'Claude 3 Sonnet', 'Balanced Claude 3 model', 200000, 4096, TRUE, TRUE, FALSE, FALSE, 3.00, 15.00),
('anthropic', 'claude-3-haiku-20240307', 'Claude 3 Haiku', 'Fastest Claude 3 model', 200000, 4096, TRUE, TRUE, FALSE, FALSE, 0.25, 1.25),
('anthropic', 'claude-3-5-sonnet-20241022', 'Claude 3.5 Sonnet', 'Latest Claude 3.5 model', 200000, 8192, TRUE, TRUE, FALSE, FALSE, 3.00, 15.00),
-- OpenAI models
('openai', 'gpt-4-turbo', 'GPT-4 Turbo', 'Latest GPT-4 Turbo with vision', 128000, 4096, TRUE, TRUE, TRUE, TRUE, 10.00, 30.00),
('openai', 'gpt-4o', 'GPT-4o', 'Multimodal GPT-4', 128000, 4096, TRUE, TRUE, TRUE, TRUE, 5.00, 15.00),
('openai', 'gpt-4o-mini', 'GPT-4o Mini', 'Small multimodal GPT-4', 128000, 16384, TRUE, TRUE, TRUE, TRUE, 0.15, 0.60),
('openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 'Fast and efficient', 16385, 4096, FALSE, TRUE, TRUE, TRUE, 0.50, 1.50),
-- Google models
('google', 'gemini-1.5-pro', 'Gemini 1.5 Pro', 'Google''s most capable model', 2097152, 8192, TRUE, TRUE, TRUE, FALSE, 1.25, 3.75),
('google', 'gemini-1.5-flash', 'Gemini 1.5 Flash', 'Fast Gemini model', 1048576, 8192, TRUE, TRUE, TRUE, FALSE, 0.075, 0.30),
('google', 'gemini-2.0-flash-exp', 'Gemini 2.0 Flash', 'Experimental Gemini 2.0', 1048576, 8192, TRUE, TRUE, TRUE, FALSE, 0.0, 0.0),
-- Ollama models (local, no cost)
('ollama', 'llama3.2:latest', 'Llama 3.2', 'Meta''s Llama 3.2', 131072, 131072, TRUE, TRUE, FALSE, FALSE, 0.0, 0.0),
('ollama', 'mistral:latest', 'Mistral', 'Mistral AI model', 32768, 32768, FALSE, TRUE, FALSE, FALSE, 0.0, 0.0),
('ollama', 'qwen2.5-coder:latest', 'Qwen 2.5 Coder', 'Coding-focused model', 32768, 32768, FALSE, TRUE, FALSE, FALSE, 0.0, 0.0);

-- =====================================================
-- COMMENTS
-- =====================================================

COMMENT ON TABLE {{tables.llm_models}} IS 'Registry of available LLM models with cost and capability information';
COMMENT ON TABLE {{tables.llm_api_calls}} IS 'Tracks all LLM API calls with origin, cost, and usage information';
COMMENT ON TABLE {{tables.usage_analytics_daily}} IS 'Daily aggregated usage statistics by origin and user';

COMMENT ON COLUMN {{tables.llm_api_calls}}.origin IS 'Name of the calling program/application (e.g., "mcp-client", "code-assistant")';
COMMENT ON COLUMN {{tables.llm_api_calls}}.id_at_origin IS 'User identifier at the origin (username, user_id, session_id, etc.)';
COMMENT ON COLUMN {{tables.llm_models}}.dollars_per_million_tokens_input IS 'Cost in dollars per million input tokens';
COMMENT ON COLUMN {{tables.llm_models}}.dollars_per_million_tokens_output IS 'Cost in dollars per million output tokens';
COMMENT ON COLUMN {{tables.llm_models}}.inactive_from IS 'Timestamp when model became inactive (NULL means currently active)';
