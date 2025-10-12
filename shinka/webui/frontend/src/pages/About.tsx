import './About.css';

export default function About() {
  return (
    <div className="about">
      <h1>About ShinkaEvolve</h1>
      <section className="about-section">
        <h2>What is ShinkaEvolve?</h2>
        <p>
          ShinkaEvolve is an open-ended program evolution framework that applies
          evolutionary computing principles to software development. The name
          &quot;Shinka&quot; comes from the Japanese word for evolution (進化).
        </p>
      </section>
      <section className="about-section">
        <h2>Key Capabilities</h2>
        <ul>
          <li>Automated program synthesis and optimization</li>
          <li>Multi-objective fitness evaluation</li>
          <li>Novelty search and quality diversity algorithms</li>
          <li>Interactive visualization of evolutionary progress</li>
          <li>Database-backed experiment tracking</li>
        </ul>
      </section>
      <section className="about-section">
        <h2>Technology Stack</h2>
        <p>
          Built with modern technologies including Python for the core evolution
          engine, React and TypeScript for the frontend interface, and SQLite
          for data persistence.
        </p>
      </section>
    </div>
  );
}
