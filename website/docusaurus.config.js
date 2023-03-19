// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");

const math = require("remark-math");
const katex = require("rehype-katex");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Recipe Recommendation",
  tagline: "Personalized Recipe Recommendation Using Heterogeneous Graphs",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://recipe.nickthegroot.com/",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  themes: ["@docusaurus/theme-mermaid"],
  markdown: {
    mermaid: true,
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
        docs: {
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
      }),
    ],
  ],

  plugins: [
    [
      "content-docs",
      /** @type {import('@docusaurus/plugin-content-docs').Options} */
      ({
        id: "ux",
        path: "ux",
        routeBasePath: "ux",

        remarkPlugins: [math],
        rehypePlugins: [katex],
        // editUrl: ({locale, versionDocsDirPath, docPath}) => {
        //   if (locale !== defaultLocale) {
        //     return `https://crowdin.com/project/docusaurus-v2/${locale}`;
        //   }
        //   return `https://github.com/facebook/docusaurus/edit/main/website/${versionDocsDirPath}/${docPath}`;
        // },
        // remarkPlugins: [npm2yarn],
        // editCurrentVersion: true,
        // sidebarPath: require.resolve('./sidebarsCommunity.js'),
        // showLastUpdateAuthor: true,
        // showLastUpdateTime: true,
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: "img/docusaurus-social-card.jpg",
      colorMode: {
        defaultMode: "light",
        disableSwitch: true,
      },
      navbar: {
        title: "Recipe Recommendation",
        logo: {
          alt: "Site Logo",
          src: "img/hamburger.svg",
        },
        items: [
          {
            type: "doc",
            docId: "intro",
            position: "left",
            label: "Project",
          },
          {
            to: "ux",
            position: "left",
            label: "UI/UX",
          },
          {
            href: "https://docs.google.com/viewer?url=https://github.com/nickthegroot/recipe-recommendation/raw/main/reports/report.pdf",
            label: "Technical Paper",
            position: "right",
          },
          {
            href: "https://github.com/nickthegroot/recipe-recommendation",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Author",
            items: [
              {
                label: "Nick DeGroot",
                href: "https://nickthegroot.com",
              },
            ],
          },
          {
            title: "Credits",
            items: [
              {
                label: "Technical Mentorship from TigerGraph",
                href: "https://www.tigergraph.com/",
              },
              {
                label: "Design Mentorship from Deyshna Pai",
                href: "https://www.deyshnapai.com/",
              },
              {
                label: "Icons from Icon8",
                href: "https://icons8.com",
              },
            ],
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
      mermaid: {
        theme: { light: "neutral", dark: "forest" },
      },
    }),

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
      crossorigin: "anonymous",
    },
  ],
};

module.exports = config;
