import './Home.css';

export default function Home() {
  return (
    <div className="home">
      <h1>Welcome to ShinkaEvolve</h1>
      <p className="home-description">
        ShinkaEvolve is an open-ended program evolution system that enables
        continuous improvement and adaptation of software systems.
      </p>
      <div className="home-features">
        <div className="feature-card">
          <h3>Evolutionary Computing</h3>
          <p>
            Harness the power of evolutionary algorithms to optimize and evolve
            your programs.
          </p>
        </div>
        <div className="feature-card">
          <h3>Adaptive Systems</h3>
          <p>
            Build systems that learn and adapt to changing requirements and
            environments.
          </p>
        </div>
        <div className="feature-card">
          <h3>Open Architecture</h3>
          <p>
            Extensible design allows for custom evolution strategies and fitness
            functions.
          </p>
        </div>
      </div>
    </div>
  );
}
