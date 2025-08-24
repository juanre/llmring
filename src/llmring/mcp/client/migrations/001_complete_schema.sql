-- MCP Client Database Schema (schema-aware for pgdbm)
-- Uses {{schema}} for functions and {{tables.*}} for table names

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Note: Schema is created by the application if missing. No search_path mutation here.

-- =====================================================
-- UTILITY FUNCTIONS
-- =====================================================

CREATE OR REPLACE FUNCTION {{schema}}.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Chat sessions (conversations)
CREATE TABLE IF NOT EXISTS {{tables.chat_sessions}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500),
    system_prompt TEXT,
    model VARCHAR(255) NOT NULL DEFAULT 'claude-3-sonnet-20240229',
    temperature FLOAT DEFAULT 0.7 CHECK (temperature >= 0 AND temperature <= 2),
    max_tokens INTEGER CHECK (max_tokens > 0),
    tool_config JSONB DEFAULT '{}',
    created_by VARCHAR(255) NOT NULL, -- Auth-server user ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages
CREATE TABLE IF NOT EXISTS {{tables.chat_messages}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES {{tables.chat_sessions}}(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- Note: LLM model registry is managed by llmbridge
-- mcp-client should query llmbridge for model information
-- =====================================================

-- =====================================================
-- MCP SERVERS REGISTRY
-- =====================================================

-- MCP servers registry
CREATE TABLE IF NOT EXISTS {{tables.mcp_servers}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    server_url VARCHAR(1000),
    base_url VARCHAR(1000) UNIQUE,
    transport_type VARCHAR(50) NOT NULL DEFAULT 'http' CHECK (transport_type IN ('stdio', 'http', 'sse', 'websocket')),
    auth_type VARCHAR(50) NOT NULL DEFAULT 'none',

    -- Configuration (JSON)
    config JSONB,
    auth_config JSONB,
    capabilities JSONB,
    metadata JSONB,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'active',
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- MCP server health history
CREATE TABLE IF NOT EXISTS {{tables.mcp_server_health_history}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    server_id UUID NOT NULL REFERENCES {{tables.mcp_servers}}(id) ON DELETE CASCADE,
    check_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL CHECK (status IN ('healthy', 'unhealthy', 'timeout', 'error')),
    response_time_ms INTEGER,
    error_message TEXT,
    capabilities_snapshot JSONB -- Snapshot of tools/resources/prompts at check time
);

-- =====================================================
-- CONVERSATION TEMPLATES (Optional feature)
-- =====================================================

-- Conversation templates for quick starts
CREATE TABLE IF NOT EXISTS {{tables.conversation_templates}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model VARCHAR(255) NOT NULL DEFAULT 'claude-3-sonnet-20240229',
    temperature FLOAT DEFAULT 0.7 CHECK (temperature >= 0 AND temperature <= 2),
    max_tokens INTEGER CHECK (max_tokens > 0),
    tool_config JSONB DEFAULT '{}',

    -- Template metadata
    category VARCHAR(100),
    tags TEXT[],
    is_public BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255) NOT NULL, -- Auth-server user ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT templates_name_user_unique UNIQUE (name, created_by)
);

-- Template usage tracking
CREATE TABLE IF NOT EXISTS {{tables.template_usage}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID NOT NULL REFERENCES {{tables.conversation_templates}}(id) ON DELETE CASCADE,
    used_by VARCHAR(255) NOT NULL, -- Auth-server user ID
    conversation_id UUID REFERENCES {{tables.chat_sessions}}(id) ON DELETE SET NULL,
    used_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- USAGE ANALYTICS
-- =====================================================

-- Usage analytics for cost tracking
CREATE TABLE IF NOT EXISTS {{tables.usage_analytics}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL, -- Auth-server user ID
    conversation_id UUID REFERENCES {{tables.chat_sessions}}(id) ON DELETE SET NULL,
    model VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,

    -- Token usage
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,

    -- Cost calculation
    input_cost DECIMAL(10,6) DEFAULT 0,
    output_cost DECIMAL(10,6) DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,

    -- Metadata
    request_type VARCHAR(50) DEFAULT 'chat' CHECK (request_type IN ('chat', 'completion', 'tool_call')),
    mcp_server_id UUID REFERENCES mcp_servers(id) ON DELETE SET NULL,
    date DATE DEFAULT CURRENT_DATE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Daily analytics rollup
CREATE TABLE IF NOT EXISTS {{tables.analytics_daily}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL, -- Auth-server user ID
    date DATE NOT NULL,

    -- Aggregated usage
    total_conversations INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,

    -- Provider breakdown
    models_used TEXT[],
    providers_used TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT analytics_daily_unique UNIQUE (user_id, date)
);

-- LLM usage tracking for quota management
CREATE TABLE IF NOT EXISTS {{tables.llm_usage}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255), -- Auth-server user ID (nullable)
    model VARCHAR(255) NOT NULL,

    -- Usage metrics
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    estimated_cost DECIMAL(10,6) DEFAULT 0,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Usage quotas (if needed)
CREATE TABLE IF NOT EXISTS {{tables.usage_quotas}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL, -- Auth-server user ID
    quota_type VARCHAR(50) NOT NULL CHECK (quota_type IN ('tokens', 'requests', 'cost')),
    quota_limit DECIMAL(15,6) NOT NULL,
    period VARCHAR(20) NOT NULL CHECK (period IN ('daily', 'weekly', 'monthly')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT usage_quotas_unique UNIQUE (user_id, quota_type, period)
);

-- =====================================================
-- FILE ATTACHMENTS (Optional feature)
-- =====================================================

-- File attachments for conversations
CREATE TABLE IF NOT EXISTS {{tables.file_attachments}} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES {{tables.chat_sessions}}(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(200),
    file_path TEXT NOT NULL, -- Path to stored file

    -- File metadata
    upload_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(255) NOT NULL, -- Auth-server user ID
    description TEXT,

    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processed', 'error')),
    extracted_text TEXT,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Chat sessions indexes
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON {{tables.chat_sessions}}(created_by);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created ON {{tables.chat_sessions}}(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON {{tables.chat_sessions}}(updated_at DESC);

-- Chat messages indexes
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON {{tables.chat_messages}}(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON {{tables.chat_messages}}(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON {{tables.chat_messages}}(role);


-- MCP servers indexes
CREATE INDEX IF NOT EXISTS idx_mcp_servers_user ON {{tables.mcp_servers}}(created_by);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_status ON {{tables.mcp_servers}}(status);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_transport ON {{tables.mcp_servers}}(transport_type);

-- MCP server health indexes
CREATE INDEX IF NOT EXISTS idx_server_health_server_id ON {{tables.mcp_server_health_history}}(server_id);
CREATE INDEX IF NOT EXISTS idx_server_health_check_time ON {{tables.mcp_server_health_history}}(check_time DESC);

-- Templates indexes
CREATE INDEX IF NOT EXISTS idx_templates_user ON {{tables.conversation_templates}}(created_by);
CREATE INDEX IF NOT EXISTS idx_templates_public ON {{tables.conversation_templates}}(is_public);
CREATE INDEX IF NOT EXISTS idx_templates_category ON {{tables.conversation_templates}}(category);

-- Template usage indexes
CREATE INDEX IF NOT EXISTS idx_template_usage_template ON {{tables.template_usage}}(template_id);
CREATE INDEX IF NOT EXISTS idx_template_usage_user ON {{tables.template_usage}}(used_by);

-- Usage analytics indexes
CREATE INDEX IF NOT EXISTS idx_usage_analytics_user ON {{tables.usage_analytics}}(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_date ON {{tables.usage_analytics}}(date);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_model ON {{tables.usage_analytics}}(model);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_conversation ON {{tables.usage_analytics}}(conversation_id);

-- Daily analytics indexes
CREATE INDEX IF NOT EXISTS idx_analytics_daily_user_date ON {{tables.analytics_daily}}(user_id, date DESC);

-- LLM usage indexes
CREATE INDEX IF NOT EXISTS idx_llm_usage_user ON {{tables.llm_usage}}(user_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model ON {{tables.llm_usage}}(model);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON {{tables.llm_usage}}(created_at);

-- Usage quotas indexes
CREATE INDEX IF NOT EXISTS idx_usage_quotas_user ON {{tables.usage_quotas}}(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_quotas_active ON {{tables.usage_quotas}}(is_active);

-- File attachments indexes
CREATE INDEX IF NOT EXISTS idx_file_attachments_session ON {{tables.file_attachments}}(session_id);
CREATE INDEX IF NOT EXISTS idx_file_attachments_user ON {{tables.file_attachments}}(uploaded_by);

-- =====================================================
-- TRIGGERS
-- =====================================================

-- Auto-update updated_at timestamps
CREATE TRIGGER tr_chat_sessions_updated_at
    BEFORE UPDATE ON {{tables.chat_sessions}}
    FOR EACH ROW EXECUTE FUNCTION {{schema}}.update_updated_at();


CREATE TRIGGER tr_mcp_servers_updated_at
    BEFORE UPDATE ON {{tables.mcp_servers}}
    FOR EACH ROW EXECUTE FUNCTION {{schema}}.update_updated_at();

CREATE TRIGGER tr_conversation_templates_updated_at
    BEFORE UPDATE ON {{tables.conversation_templates}}
    FOR EACH ROW EXECUTE FUNCTION {{schema}}.update_updated_at();


CREATE TRIGGER tr_usage_quotas_updated_at
    BEFORE UPDATE ON {{tables.usage_quotas}}
    FOR EACH ROW EXECUTE FUNCTION {{schema}}.update_updated_at();

-- =====================================================
-- VIEWS FOR CONVENIENCE
-- =====================================================
-- Note: Views are commented out as they don't support schema placeholders
-- They should be created separately if needed

-- Conversation summary view
-- CREATE VIEW conversation_summary AS
-- SELECT
--     cs.id,
--     cs.title,
--     cs.model,
--     cs.created_by,
--     cs.created_at,
--     cs.updated_at,
--     COUNT(cm.id) as message_count,
--     COALESCE(SUM(cm.token_count), 0) as total_tokens,
--     (
--         SELECT content
--         FROM chat_messages
--         WHERE session_id = cs.id
--         ORDER BY timestamp DESC
--         LIMIT 1
--     ) as last_message_preview
-- FROM chat_sessions cs
-- LEFT JOIN chat_messages cm ON cs.id = cm.session_id
-- GROUP BY cs.id, cs.title, cs.model, cs.created_by, cs.created_at, cs.updated_at;
--
-- -- User analytics summary view
-- CREATE VIEW user_analytics_summary AS
-- SELECT
--     user_id,
--     COUNT(DISTINCT conversation_id) as total_conversations,
--     COUNT(*) as total_requests,
--     SUM(total_tokens) as total_tokens,
--     SUM(total_cost) as total_cost,
--     array_agg(DISTINCT model) as models_used,
--     array_agg(DISTINCT provider) as providers_used,
--     MIN(date) as first_usage,
--     MAX(date) as last_usage
-- FROM usage_analytics
-- GROUP BY user_id;
--
-- -- =====================================================
-- -- INITIAL DATA
-- -- =====================================================
--
-- -- Pre-populate LLM models registry with common models
-- INSERT INTO llm_models (model_key, display_name, provider, model_family, max_tokens, supports_streaming, supports_tools, supports_vision, supports_json_mode, input_cost_per_million, output_cost_per_million, context_window, description) VALUES
-- -- OpenAI Models
-- ('gpt-4o', 'GPT-4o', 'openai', 'gpt-4', 4096, true, true, true, true, 5.00, 15.00, 128000, 'OpenAI GPT-4o multimodal model'),
-- ('gpt-4o-mini', 'GPT-4o Mini', 'openai', 'gpt-4', 16384, true, true, true, true, 0.15, 0.60, 128000, 'OpenAI GPT-4o Mini - cost-effective model'),
-- ('gpt-4-turbo', 'GPT-4 Turbo', 'openai', 'gpt-4', 4096, true, true, true, true, 10.00, 30.00, 128000, 'OpenAI GPT-4 Turbo'),
-- ('gpt-3.5-turbo', 'GPT-3.5 Turbo', 'openai', 'gpt-3.5', 4096, true, true, false, true, 0.50, 1.50, 16385, 'OpenAI GPT-3.5 Turbo'),
--
-- -- Anthropic Models
-- ('claude-3-5-sonnet-20241022', 'Claude 3.5 Sonnet', 'anthropic', 'claude-3', 8192, true, true, true, false, 3.00, 15.00, 200000, 'Anthropic Claude 3.5 Sonnet'),
-- ('claude-3-5-haiku-20241022', 'Claude 3.5 Haiku', 'anthropic', 'claude-3', 8192, true, true, true, false, 1.00, 5.00, 200000, 'Anthropic Claude 3.5 Haiku'),
-- ('claude-3-opus-20240229', 'Claude 3 Opus', 'anthropic', 'claude-3', 4096, true, true, true, false, 15.00, 75.00, 200000, 'Anthropic Claude 3 Opus'),
-- ('claude-3-sonnet-20240229', 'Claude 3 Sonnet', 'anthropic', 'claude-3', 4096, true, true, true, false, 3.00, 15.00, 200000, 'Anthropic Claude 3 Sonnet'),
-- ('claude-3-haiku-20240307', 'Claude 3 Haiku', 'anthropic', 'claude-3', 4096, true, true, true, false, 0.25, 1.25, 200000, 'Anthropic Claude 3 Haiku'),
--
-- -- Google Models
-- ('gemini-1.5-pro', 'Gemini 1.5 Pro', 'google', 'gemini', 8192, true, true, true, true, 3.50, 10.50, 2000000, 'Google Gemini 1.5 Pro'),
-- ('gemini-1.5-flash', 'Gemini 1.5 Flash', 'google', 'gemini', 8192, true, true, true, true, 0.075, 0.30, 1000000, 'Google Gemini 1.5 Flash'),
-- ('gemini-pro', 'Gemini Pro', 'google', 'gemini', 8192, true, true, false, true, 0.50, 1.50, 30720, 'Google Gemini Pro');
--
-- -- Create default conversation template
-- INSERT INTO conversation_templates (name, description, system_prompt, category, is_public, created_by) VALUES
-- ('General Assistant', 'A helpful general-purpose AI assistant', 'You are a helpful AI assistant. Provide accurate, helpful, and thoughtful responses to user questions and requests.', 'general', true, 'system'),
-- ('Code Assistant', 'AI assistant specialized for programming agents', 'You are an expert programming assistant. Help users write, debug, and explain code. Provide clear explanations and best practices.', 'programming', true, 'system'),
-- ('Writing Assistant', 'AI assistant for writing and editing agents', 'You are a skilled writing assistant. Help users improve their writing, check grammar, suggest improvements, and maintain their voice and style.', 'writing', true, 'system');
