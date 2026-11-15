import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import {
  BarChart3,
  Layers,
  AlertTriangle,
  Zap,
  Info,
  CheckCircle2,
  XCircle,
  Clock,
  Coins,
  ShieldCheck,
  Search,
} from 'lucide-react';
import { DEFAULT_BENCHMARK_DATA } from './data';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedSetup, setSelectedSetup] = useState('D');
  const [promptFilter, setPromptFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);
  const pageSize = 5;

  const data = DEFAULT_BENCHMARK_DATA;

  // Overview Bar Chart Data
  const chartData = data.setups.map((s) => ({
    name: s.id,
    label: s.label,
    accuracy: Number((s.tool_accuracy * 100).toFixed(1)),
    strictMatch: Number((s.args_strict_match_rate * 100).toFixed(1)),
    tokens: s.total_tokens,
    latency: s.avg_latency_ms,
    color: s.color,
  }));

  // Filtered Prompts
  const filteredPrompts = data.sample_prompts.filter((p) => {
    const matchesSearch =
      p.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.expected_tool.toLowerCase().includes(searchQuery.toLowerCase());
    if (!matchesSearch) return false;
    if (promptFilter === 'pass') return selectedSetup === 'A' ? p.setup_a_pass : p.setup_d_pass;
    if (promptFilter === 'fail') return selectedSetup === 'A' ? !p.setup_a_pass : !p.setup_d_pass;
    return true;
  });

  const paginatedPrompts = filteredPrompts.slice((page - 1) * pageSize, page * pageSize);
  const totalPages = Math.ceil(filteredPrompts.length / pageSize) || 1;

  return (
    <div style={styles.container}>
      {/* Top Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div>
            <div style={styles.badgeRow}>
              <span style={styles.badge}>Factorial Benchmark</span>
              <span style={styles.hashBadge}>
                <ShieldCheck size={12} style={{ marginRight: 4 }} />
                SHA: {data.heldout_hash.slice(0, 10)}...
              </span>
            </div>
            <h1 style={styles.title}>Toolbench Experimental Evaluation</h1>
            <p style={styles.subtitle}>
              Measuring LLM Tool Selection on MCP: Retrieval vs Description Rewriting ($2 \times 2$ Factorial Design)
            </p>
          </div>
          <div style={styles.headerMeta}>
            <div style={styles.metaItem}>
              <span style={styles.metaLabel}>Model</span>
              <span style={styles.metaValue}>{data.model}</span>
            </div>
            <div style={styles.metaItem}>
              <span style={styles.metaLabel}>Eval Temperature</span>
              <span style={styles.metaValue}>0.0 (Strictly Deterministic)</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={styles.tabNav}>
          {[
            { id: 'overview', label: 'Overview & Statistical Tests', icon: BarChart3 },
            { id: 'details', label: 'Setup Detail & Prompts', icon: Layers },
            { id: 'failures', label: 'Failure Analysis (F9)', icon: AlertTriangle },
            { id: 'efficiency', label: 'Token & Latency Efficiency', icon: Zap },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  ...styles.tabButton,
                  ...(isActive ? styles.activeTabButton : {}),
                }}
              >
                <Icon size={16} style={{ marginRight: 8 }} />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </header>

      {/* Main Content Area */}
      <main style={styles.main}>
        {/* =========================================================================
            TAB 1: OVERVIEW & STATISTICAL COMPARISON
           ========================================================================= */}
        {activeTab === 'overview' && (
          <div style={styles.tabContent}>
            {/* Factorial Metric Cards */}
            <div style={styles.cardsGrid}>
              {data.setups.map((s) => (
                <div key={s.id} style={styles.card}>
                  <div style={styles.cardHeader}>
                    <span style={{ ...styles.setupTag, color: s.color, borderColor: s.color }}>
                      {s.id}
                    </span>
                    <span style={styles.cardMetricValue}>
                      {(s.tool_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <h3 style={styles.cardTitle}>{s.label.split('(')[0]}</h3>
                  <p style={styles.cardDescription}>{s.description}</p>
                  <div style={styles.cardFooter}>
                    <div style={styles.subMetric}>
                      <span style={styles.subMetricLabel}>Strict Args:</span>
                      <span style={styles.subMetricValue}>
                        {(s.args_strict_match_rate * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div style={styles.subMetric}>
                      <span style={styles.subMetricLabel}>Post-Retry:</span>
                      <span style={styles.subMetricValue}>
                        {(s.args_valid_post_retry_rate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Primary Accuracy Chart */}
            <div style={styles.sectionPanel}>
              <div style={styles.panelHeader}>
                <div>
                  <h2 style={styles.panelTitle}>Tool Selection Accuracy by Experimental Setup</h2>
                  <p style={styles.panelSubtitle}>
                    Evaluating primary outcome across 50 frozen held-out prompts ($T=0.0$)
                  </p>
                </div>
              </div>
              <div style={{ height: 260, width: '100%', marginTop: 16 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                    <XAxis
                      dataKey="name"
                      stroke="var(--text-tertiary)"
                      tick={{ fill: 'var(--text-secondary)' }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      stroke="var(--text-tertiary)"
                      tick={{ fill: 'var(--text-secondary)' }}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--bg-secondary)',
                        borderColor: 'var(--border-muted)',
                        borderRadius: 8,
                        color: 'var(--text-primary)',
                      }}
                      formatter={(val) => [`${val}%`, 'Accuracy']}
                    />
                    <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Statistical Hypothesis Tests Table */}
            <div style={styles.sectionPanel}>
              <div style={styles.panelHeader}>
                <div>
                  <h2 style={styles.panelTitle}>Hypothesis Testing & Multiplicity Control</h2>
                  <p style={styles.panelSubtitle}>
                    Paired McNemar tests with exact binomial p-values, 95% bootstrap CIs, and step-down Holm-Bonferroni correction
                  </p>
                </div>
              </div>

              <div style={styles.tableWrapper}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Comparison</th>
                      <th style={styles.th}>Baseline vs Treatment</th>
                      <th style={styles.thRight}>Delta ($\Delta$)</th>
                      <th style={styles.thCenter}>95% Bootstrap CI</th>
                      <th style={styles.thRight}>Raw p-value</th>
                      <th style={styles.thRight}>Holm Adj. p</th>
                      <th style={styles.thCenter}>Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.comparisons.map((c, i) => (
                      <tr key={i} style={styles.tr}>
                        <td style={styles.tdBold}>{c.comparison}</td>
                        <td style={styles.tdSecondary}>
                          {c.baseline} &rarr; {c.treatment}
                        </td>
                        <td style={{ ...styles.tdRight, color: 'var(--status-pass-text)' }}>
                          {c.delta}
                        </td>
                        <td style={styles.tdCenterMono}>{c.ci_95}</td>
                        <td style={styles.tdRightMono}>{c.raw_p.toFixed(4)}</td>
                        <td style={{ ...styles.tdRightMono, color: 'var(--accent-muted)' }}>
                          {c.holm_p !== null ? c.holm_p.toFixed(4) : '&mdash;'}
                        </td>
                        <td style={styles.tdCenter}>
                          <span
                            style={{
                              ...styles.statusPill,
                              backgroundColor: c.is_primary
                                ? 'var(--status-pass-bg)'
                                : 'var(--accent-subtle)',
                              color: c.is_primary
                                ? 'var(--status-pass-text)'
                                : 'var(--accent-muted)',
                            }}
                          >
                            {c.significance}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={styles.infoBox}>
                <Info size={18} style={{ color: 'var(--accent-muted)', marginRight: 12, flexShrink: 0 }} />
                <p style={styles.infoText}>
                  <strong>Methodological Multiplicity Note:</strong> When evaluating multiple simultaneous hypotheses, the probability of false-positive discoveries accumulates. Step-down Holm-Bonferroni correction controls the family-wise error rate across primary claims ($A \text{ vs } C$, $A \text{ vs } B$, $A \text{ vs } D$) while preserving statistical power.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* =========================================================================
            TAB 2: SETUP DETAIL & PER-PROMPT INSPECTION
           ========================================================================= */}
        {activeTab === 'details' && (
          <div style={styles.tabContent}>
            {/* Setup Selector */}
            <div style={styles.setupSelectorBar}>
              <span style={styles.selectorLabel}>Select Experimental Setup:</span>
              <div style={styles.buttonGroup}>
                {data.setups.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setSelectedSetup(s.id)}
                    style={{
                      ...styles.selectorButton,
                      ...(selectedSetup === s.id ? styles.activeSelectorButton : {}),
                    }}
                  >
                    Setup {s.id}
                  </button>
                ))}
              </div>
            </div>

            {/* Filter and Search Bar */}
            <div style={styles.filterBar}>
              <div style={styles.searchBox}>
                <Search size={16} style={{ color: 'var(--text-tertiary)', marginRight: 8 }} />
                <input
                  type="text"
                  placeholder="Search prompt queries or expected tools..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={styles.searchInput}
                />
              </div>

              <div style={styles.filterButtonGroup}>
                {['all', 'pass', 'fail'].map((f) => (
                  <button
                    key={f}
                    onClick={() => {
                      setPromptFilter(f);
                      setPage(1);
                    }}
                    style={{
                      ...styles.filterButton,
                      ...(promptFilter === f ? styles.activeFilterButton : {}),
                    }}
                  >
                    {f.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Prompt Table */}
            <div style={styles.sectionPanel}>
              <div style={styles.tableWrapper}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Prompt ID</th>
                      <th style={styles.th}>User Query</th>
                      <th style={styles.th}>Expected Tool</th>
                      <th style={styles.th}>Selected Tool ({selectedSetup})</th>
                      <th style={styles.thCenter}>Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedPrompts.map((p) => {
                      const isPass = selectedSetup === 'A' ? p.setup_a_pass : p.setup_d_pass;
                      const chosenTool = selectedSetup === 'A' ? p.setup_a_chosen : p.setup_d_chosen;
                      return (
                        <tr key={p.id} style={styles.tr}>
                          <td style={styles.tdMono}>{p.id}</td>
                          <td style={styles.td}>{p.query}</td>
                          <td style={styles.tdToolExpected}>{p.expected_tool}</td>
                          <td style={styles.tdToolChosen}>{chosenTool}</td>
                          <td style={styles.tdCenter}>
                            {isPass ? (
                              <span style={styles.passBadge}>
                                <CheckCircle2 size={13} style={{ marginRight: 4 }} /> Pass
                              </span>
                            ) : (
                              <span style={styles.failBadge}>
                                <XCircle size={13} style={{ marginRight: 4 }} /> Fail
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div style={styles.paginationBar}>
                <span style={styles.pageInfo}>
                  Showing page {page} of {totalPages} ({filteredPrompts.length} prompts)
                </span>
                <div style={styles.paginationButtons}>
                  <button
                    disabled={page <= 1}
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    style={styles.pageButton}
                  >
                    Previous
                  </button>
                  <button
                    disabled={page >= totalPages}
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    style={styles.pageButton}
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* =========================================================================
            TAB 3: FAILURE MODE BREAKDOWN (F9)
           ========================================================================= */}
        {activeTab === 'failures' && (
          <div style={styles.tabContent}>
            <div style={styles.sectionPanel}>
              <div style={styles.panelHeader}>
                <div>
                  <h2 style={styles.panelTitle}>Hierarchical Failure Mode Breakdown (F9)</h2>
                  <p style={styles.panelSubtitle}>
                    Priority-ordered categorization across non-passing prompts (First match wins)
                  </p>
                </div>
              </div>

              <div style={styles.failureGrid}>
                {/* Failure Category Cards */}
                <div style={styles.failureList}>
                  {data.failure_breakdown.map((item, idx) => (
                    <div key={idx} style={styles.failureRow}>
                      <div style={styles.failureRowHeader}>
                        <span style={{ ...styles.categoryBullet, backgroundColor: item.color }} />
                        <span style={styles.categoryName}>{item.category}</span>
                      </div>
                      <div style={styles.failureRowStats}>
                        <span style={styles.categoryCount}>{item.count}</span>
                        <span style={styles.categoryPct}>({item.percentage}%)</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Pie Chart Visualization */}
                <div style={{ height: 240, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={data.failure_breakdown}
                        dataKey="count"
                        nameKey="category"
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={85}
                        paddingAngle={3}
                      >
                        {data.failure_breakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--bg-secondary)',
                          borderColor: 'var(--border-muted)',
                          borderRadius: 8,
                          color: 'var(--text-primary)',
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div style={styles.priorityBox}>
                <h4 style={styles.priorityTitle}>Priority Evaluation Cascade</h4>
                <ol style={styles.priorityList}>
                  <li><strong>1. Error:</strong> LLM timeouts, network failures, or API rate exhaustion.</li>
                  <li><strong>2. Retrieval Miss:</strong> Target tool omitted from top-k retrieved candidates (Setups C & D).</li>
                  <li><strong>3. No Tool Call:</strong> Model produced conversational text without emitting a function call.</li>
                  <li><strong>4. Confusion:</strong> Target tool was exposed in context, but model selected an alternative tool.</li>
                  <li><strong>5. Invalid Arguments (Unrecovered):</strong> Argument schema validation failed even after single-turn retry.</li>
                  <li><strong>6. Wrong Arguments:</strong> Schema valid, but failed semantic equality check against acceptable args.</li>
                </ol>
              </div>
            </div>
          </div>
        )}

        {/* =========================================================================
            TAB 4: TOKEN & LATENCY EFFICIENCY
           ========================================================================= */}
        {activeTab === 'efficiency' && (
          <div style={styles.tabContent}>
            <div style={styles.cardsGrid}>
              <div style={styles.card}>
                <div style={styles.cardHeader}>
                  <Coins size={20} style={{ color: 'var(--accent-muted)' }} />
                  <span style={styles.cardMetricValue}>-60.2%</span>
                </div>
                <h3 style={styles.cardTitle}>Context Token Reduction</h3>
                <p style={styles.cardDescription}>
                  Retrieval (Setups C & D) reduces exposed prompt context from 38.2k tokens to 15.2k tokens across the evaluation batch.
                </p>
              </div>

              <div style={styles.card}>
                <div style={styles.cardHeader}>
                  <Clock size={20} style={{ color: 'var(--accent-muted)' }} />
                  <span style={styles.cardMetricValue}>-38.7%</span>
                </div>
                <h3 style={styles.cardTitle}>Inference Latency Delta</h3>
                <p style={styles.cardDescription}>
                  Narrowing the exposed schema set drops average response latency from 124ms to 76ms per invocation.
                </p>
              </div>
            </div>

            <div style={styles.sectionPanel}>
              <h2 style={styles.panelTitle}>Tokens per Benchmark Batch</h2>
              <div style={{ height: 240, width: '100%', marginTop: 16 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                    <XAxis dataKey="name" stroke="var(--text-tertiary)" tick={{ fill: 'var(--text-secondary)' }} />
                    <YAxis stroke="var(--text-tertiary)" tick={{ fill: 'var(--text-secondary)' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--bg-secondary)',
                        borderColor: 'var(--border-muted)',
                        borderRadius: 8,
                        color: 'var(--text-primary)',
                      }}
                      formatter={(val) => [`${val.toLocaleString()} tokens`, 'Total Tokens']}
                    />
                    <Bar dataKey="tokens" radius={[6, 6, 0, 0]} fill="var(--accent-muted)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Minimal, Dark Theme Styles using CSS Variables from tokens.css
const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-sans)',
  },
  header: {
    backgroundColor: 'var(--bg-secondary)',
    borderBottom: '1px solid var(--border-subtle)',
    padding: '24px 32px 0 32px',
  },
  headerContent: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    gap: '16px',
    marginBottom: '20px',
  },
  badgeRow: {
    display: 'flex',
    gap: '8px',
    marginBottom: '8px',
    alignItems: 'center',
  },
  badge: {
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--accent-muted)',
    backgroundColor: 'var(--accent-subtle)',
    padding: '2px 8px',
    borderRadius: 'var(--radius-sm)',
  },
  hashBadge: {
    fontSize: '11px',
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-secondary)',
    backgroundColor: 'var(--bg-tertiary)',
    padding: '2px 8px',
    borderRadius: 'var(--radius-sm)',
    display: 'inline-flex',
    alignItems: 'center',
  },
  title: {
    fontSize: '22px',
    fontWeight: 700,
    margin: 0,
    color: 'var(--text-primary)',
    letterSpacing: '-0.02em',
  },
  subtitle: {
    fontSize: '13px',
    color: 'var(--text-secondary)',
    margin: '4px 0 0 0',
  },
  headerMeta: {
    display: 'flex',
    gap: '24px',
    alignItems: 'center',
  },
  metaItem: {
    display: 'flex',
    flexDirection: 'column',
  },
  metaLabel: {
    fontSize: '11px',
    textTransform: 'uppercase',
    color: 'var(--text-tertiary)',
    fontWeight: 600,
  },
  metaValue: {
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--text-primary)',
    marginTop: '2px',
  },
  tabNav: {
    display: 'flex',
    gap: '8px',
  },
  tabButton: {
    backgroundColor: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    color: 'var(--text-secondary)',
    padding: '12px 16px',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    display: 'inline-flex',
    alignItems: 'center',
    transition: 'all 0.15s ease',
  },
  activeTabButton: {
    color: 'var(--accent-muted)',
    borderBottom: '2px solid var(--accent-muted)',
  },
  main: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '32px 24px',
  },
  tabContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  cardsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
    gap: '16px',
  },
  card: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)',
    padding: '20px',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  setupTag: {
    fontSize: '13px',
    fontWeight: 700,
    border: '1px solid',
    padding: '2px 8px',
    borderRadius: 'var(--radius-sm)',
  },
  cardMetricValue: {
    fontSize: '26px',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.02em',
  },
  cardTitle: {
    fontSize: '14px',
    fontWeight: 600,
    margin: '0 0 6px 0',
    color: 'var(--text-primary)',
  },
  cardDescription: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
    margin: 0,
    lineHeight: '1.5',
    minHeight: '36px',
  },
  cardFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: '16px',
    paddingTop: '12px',
    borderTop: '1px solid var(--border-subtle)',
  },
  subMetric: {
    display: 'flex',
    gap: '6px',
    fontSize: '12px',
  },
  subMetricLabel: {
    color: 'var(--text-tertiary)',
  },
  subMetricValue: {
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  sectionPanel: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)',
    padding: '24px',
  },
  panelHeader: {
    marginBottom: '16px',
  },
  panelTitle: {
    fontSize: '16px',
    fontWeight: 600,
    margin: 0,
    color: 'var(--text-primary)',
  },
  panelSubtitle: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
    margin: '4px 0 0 0',
  },
  tableWrapper: {
    overflowX: 'auto',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    textAlign: 'left',
    fontSize: '13px',
  },
  th: {
    padding: '10px 12px',
    color: 'var(--text-tertiary)',
    fontWeight: 600,
    borderBottom: '1px solid var(--border-muted)',
    fontSize: '11px',
    textTransform: 'uppercase',
  },
  thRight: {
    padding: '10px 12px',
    color: 'var(--text-tertiary)',
    fontWeight: 600,
    borderBottom: '1px solid var(--border-muted)',
    fontSize: '11px',
    textTransform: 'uppercase',
    textAlign: 'right',
  },
  thCenter: {
    padding: '10px 12px',
    color: 'var(--text-tertiary)',
    fontWeight: 600,
    borderBottom: '1px solid var(--border-muted)',
    fontSize: '11px',
    textTransform: 'uppercase',
    textAlign: 'center',
  },
  tr: {
    borderBottom: '1px solid var(--border-subtle)',
  },
  td: {
    padding: '12px',
    color: 'var(--text-primary)',
  },
  tdBold: {
    padding: '12px',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  tdSecondary: {
    padding: '12px',
    color: 'var(--text-secondary)',
  },
  tdRight: {
    padding: '12px',
    textAlign: 'right',
    fontWeight: 600,
  },
  tdRightMono: {
    padding: '12px',
    textAlign: 'right',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
  },
  tdCenter: {
    padding: '12px',
    textAlign: 'center',
  },
  tdCenterMono: {
    padding: '12px',
    textAlign: 'center',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
  },
  tdMono: {
    padding: '12px',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--text-secondary)',
  },
  tdToolExpected: {
    padding: '12px',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--accent-muted)',
  },
  tdToolChosen: {
    padding: '12px',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--text-secondary)',
  },
  statusPill: {
    display: 'inline-block',
    padding: '3px 8px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '11px',
    fontWeight: 600,
  },
  infoBox: {
    display: 'flex',
    alignItems: 'flex-start',
    backgroundColor: 'var(--accent-subtle)',
    border: '1px solid rgba(56, 189, 248, 0.2)',
    borderRadius: 'var(--radius-md)',
    padding: '14px 16px',
    marginTop: '20px',
  },
  infoText: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
    margin: 0,
    lineHeight: '1.6',
  },
  setupSelectorBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  selectorLabel: {
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--text-secondary)',
  },
  buttonGroup: {
    display: 'flex',
    gap: '6px',
  },
  selectorButton: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-subtle)',
    color: 'var(--text-secondary)',
    padding: '6px 14px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '12px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  activeSelectorButton: {
    backgroundColor: 'var(--accent-subtle)',
    borderColor: 'var(--accent-muted)',
    color: 'var(--accent-muted)',
  },
  filterBar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '16px',
    flexWrap: 'wrap',
  },
  searchBox: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-sm)',
    padding: '8px 12px',
    flex: 1,
    minWidth: '240px',
  },
  searchInput: {
    backgroundColor: 'transparent',
    border: 'none',
    outline: 'none',
    color: 'var(--text-primary)',
    fontSize: '13px',
    width: '100%',
  },
  filterButtonGroup: {
    display: 'flex',
    gap: '6px',
  },
  filterButton: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-subtle)',
    color: 'var(--text-tertiary)',
    padding: '6px 12px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '11px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  activeFilterButton: {
    backgroundColor: 'var(--bg-tertiary)',
    color: 'var(--text-primary)',
    borderColor: 'var(--border-active)',
  },
  passBadge: {
    color: 'var(--status-pass-text)',
    backgroundColor: 'var(--status-pass-bg)',
    padding: '2px 8px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '11px',
    fontWeight: 600,
    display: 'inline-flex',
    alignItems: 'center',
  },
  failBadge: {
    color: 'var(--status-fail-text)',
    backgroundColor: 'var(--status-fail-bg)',
    padding: '2px 8px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '11px',
    fontWeight: 600,
    display: 'inline-flex',
    alignItems: 'center',
  },
  paginationBar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '16px',
    paddingTop: '12px',
    borderTop: '1px solid var(--border-subtle)',
  },
  pageInfo: {
    fontSize: '12px',
    color: 'var(--text-tertiary)',
  },
  paginationButtons: {
    display: 'flex',
    gap: '8px',
  },
  pageButton: {
    backgroundColor: 'var(--bg-tertiary)',
    border: '1px solid var(--border-subtle)',
    color: 'var(--text-secondary)',
    padding: '4px 12px',
    borderRadius: 'var(--radius-sm)',
    fontSize: '12px',
    cursor: 'pointer',
  },
  failureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '24px',
    alignItems: 'center',
  },
  failureList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  failureRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 14px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: 'var(--radius-sm)',
  },
  failureRowHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  categoryBullet: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  categoryName: {
    fontSize: '13px',
    color: 'var(--text-primary)',
    fontWeight: 500,
  },
  failureRowStats: {
    display: 'flex',
    gap: '6px',
    alignItems: 'baseline',
  },
  categoryCount: {
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  categoryPct: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
  },
  priorityBox: {
    marginTop: '24px',
    paddingTop: '16px',
    borderTop: '1px solid var(--border-subtle)',
  },
  priorityTitle: {
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--text-secondary)',
    margin: '0 0 8px 0',
  },
  priorityList: {
    fontSize: '12px',
    color: 'var(--text-tertiary)',
    paddingLeft: '18px',
    margin: 0,
    lineHeight: '1.8',
  },
};
