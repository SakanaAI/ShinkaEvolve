# ShinkaEvolve WebUI (React)

Modern React-based web interface for visualizing and analyzing ShinkaEvolve optimization results.

## Features

### Left Panel Views
- **Tree View**: Interactive D3.js genealogy tree showing program evolution
- **Programs Table**: Sortable, filterable table of all programs with metrics
- **Metrics View**: Performance metrics visualization over generations
- **Embeddings View**: Code embedding similarity heatmap
- **Clusters View**: 2D/3D PCA visualization of program clusters
- **Islands View**: Multi-island evolution tracking
- **LLM Posterior**: Model selection tracking over time
- **Path to Best**: Visualization of lineage to best program

### Right Panel Details
- **Meta**: Program metadata and key information
- **Pareto Front**: Multi-objective optimization visualization
- **Scratchpad**: LLM meta-recommendations and analysis
- **Node Details**: Complete program information
- **Code**: Syntax-highlighted code viewer with copy/download
- **Diff**: Code changes visualization
- **Evaluation**: Execution logs and feedback
- **LLM Result**: Generation details and API costs

### Key Features
- Real-time data updates with configurable auto-refresh
- Resizable split-panel layout
- Database and task selection
- Performance-optimized rendering
- TypeScript type safety
- Modern, responsive UI

## Prerequisites

- Node.js >= 18
- npm or yarn
- Python backend running ShinkaEvolve visualization server

## Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The development server will start on `http://localhost:3000` and proxy API requests to `http://localhost:8000` (the Python backend).

### Project Structure

```
src/
├── components/          # React components
│   ├── views/          # Left panel view components
│   ├── details/        # Right panel detail components
│   ├── Header.tsx      # Top navigation bar
│   ├── LeftPanel.tsx   # Left panel container
│   └── RightPanel.tsx  # Right panel container
├── hooks/              # Custom React hooks
│   └── usePrograms.ts  # Data fetching hooks
├── lib/                # Utilities
│   ├── api.ts          # API client
│   └── utils.ts        # Helper functions
├── store/              # State management
│   └── useStore.ts     # Zustand store
├── types/              # TypeScript types
│   └── program.ts      # Data type definitions
├── App.tsx             # Main app component
├── main.tsx            # App entry point
└── index.css           # Global styles
```

## Usage with ShinkaEvolve

1. Start your ShinkaEvolve visualization server:
```bash
shinka_visualize /path/to/results --port 8000
```

2. In a separate terminal, start the React dev server:
```bash
cd webui-react
npm run dev
```

3. Open `http://localhost:3000` in your browser

## Configuration

### API Proxy

The Vite development server is configured to proxy API requests to `http://localhost:8000`. To change this, edit `vite.config.ts`:

```typescript
server: {
  port: 3000,
  proxy: {
    '/list_databases': 'http://localhost:8000',
    '/get_programs': 'http://localhost:8000',
    // ... other endpoints
  },
}
```

### Auto-refresh Interval

The default auto-refresh interval is 3 seconds. To change this, edit `src/store/useStore.ts`:

```typescript
autoRefreshInterval: 3000, // milliseconds
```

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TanStack Query** - Data fetching and caching
- **Zustand** - State management
- **D3.js** - Data visualization
- **Recharts** - Chart components
- **TailwindCSS** - Styling
- **react-resizable-panels** - Resizable layout
- **Lucide React** - Icons

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory. Serve them with any static file server:

```bash
npm run preview
```

Or deploy to your preferred hosting platform.

## Future Enhancements

- [ ] Complete Metrics view with Recharts
- [ ] Implement Embeddings heatmap with D3
- [ ] Add 3D cluster visualization
- [ ] Implement Pareto front visualization
- [ ] Add advanced diff viewer with syntax highlighting
- [ ] Theme switching (light/dark mode)
- [ ] Export visualizations as images
- [ ] Keyboard shortcuts
- [ ] Search and filter functionality
- [ ] Program comparison view

## Contributing

This is part of the ShinkaEvolve project. For contribution guidelines, see the main repository README.

## License

Apache 2.0 - See LICENSE file in the main repository.
