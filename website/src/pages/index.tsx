import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import clsx from "clsx";

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
            to="https://github.com/nickthegroot/recipe-recommendation/raw/main/reports/report.pdf"
          >
            🔗 Read the Paper
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
      <main>{/* <HomepageFeatures /> */}</main>
    </Layout>
  );
}
