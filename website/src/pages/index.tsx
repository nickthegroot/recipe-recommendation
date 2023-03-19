import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import clsx from "clsx";
import React from "react";
import HomepageProblems from "../components/HomepageProblems";
import HomescreenContainer from "../components/HomescreenContainer";
import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">
          <i>
            "Personalized Recipe Recommendation Using Heterogeneous Graphs"
            (2023)
          </i>
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
            style={{ margin: 10 }}
          >
            🔗 Learn About The Project
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/ux"
            style={{ margin: 10 }}
          >
            🔗 Learn About The Design
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="Homepage"
      description={`Homepage for the ${siteConfig.title} project`}
    >
      <HomepageHeader />
      <br />
      <main>
        <HomepageProblems />
        <HomescreenContainer>
          <h1>The Solution</h1>
          <p style={{ width: "75%", textAlign: "center" }}>
            What's needed is a way to find new recipes that you'll actually
            enjoy, personalized to the things you already have on hand. To this
            end, we present the first steps in making this dream a reality: the{" "}
            <b>recipe recommendation engine</b> and the <b>user experience</b>.
          </p>
        </HomescreenContainer>
      </main>
    </Layout>
  );
}
