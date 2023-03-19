import React from "react";
import clsx from "clsx";
import styles from "./styles.module.css";

type ProblemItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: JSX.Element;
};

const ProblemList: ProblemItem[] = [
  {
    Svg: require("@site/static/img/icons/repeat.svg").default,
    title: "Eating In Can Get Repetitive",
    description: (
      <>
        Meals can often feel like you've been eating the same thing for weeks.
      </>
    ),
  },
  {
    Svg: require("@site/static/img/icons/dollar.svg").default,
    title: "Eating Out Can Get Expensive",
    description: (
      <>
        Ordering food at a restaurant can rack up quick, particularly for
        families.
      </>
    ),
  },
  {
    Svg: require("@site/static/img/icons/heart.svg").default,
    title: "Ordering In Can Get Unhealthy",
    description: (
      <>
        With the wide variety of food delivered straight to your door, you're
        not always incentivized to go for the healthiest option
      </>
    ),
  },
];

function Problem({ title, Svg, description }: ProblemItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageProblems(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center">
          <h1>The Problem</h1>
        </div>
        <br />
        <div className="row">
          {ProblemList.map((props, idx) => (
            <Problem key={idx} {...props} />
          ))}
        </div>
        <div className="text--center">
          In essence,{" "}
          <i>
            <b>
              busy adults need a way to find tasty yet healthy recipes that
              outweigh the convenience and simplicity of ordering out.
            </b>
          </i>
        </div>
      </div>
    </section>
  );
}
