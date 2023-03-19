# Prototyping

## Wireframes

<iframe width="100%" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FKZC57n5PAwZ1SGFazgWxyx%2FWireframe%3Fnode-id%3D3507%253A2917%26t%3DaxFg2d2VI2U1QKAp-1" allowfullscreen></iframe>

Based on the sketches presented in [Early Ideation](2-early-ideation.md), we went through and wireframed screens inside Figma. The focus of this exercise was to get the _overall layout_ of the design figured out.

### User Testing

After creating the wireframes, we mocked how the screens would work together using Figma's built in prototyping tools.

<div align="center">
<iframe
    style={{
        maxWidth: '100%',
        height: 'min(70vh, 700px)',
        aspectRatio: '390 / 844'
    }}
    src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Fproto%2FKZC57n5PAwZ1SGFazgWxyx%2FWireframe%3Fpage-id%3D3489%253A6008%26node-id%3D3496-6204%26viewport%3D587%252C702%252C0.45%26scaling%3Dscale-down%26starting-point-node-id%3D3496%253A6204%26show-proto-sidebar%3D0%26hide-ui%3D1"
    allowfullscreen
></iframe>
</div>

This prototype was then sent to users over the course of a structured session. Users were asked to complete two tasks, one after the other.

1. Find the instructions for how to cook Tuesday's meal
2. Find out how much groceries for this week will cost

Our participants were very similar to the ones we had in [user research section](1-user-research.md): a male and female both in their fifties and a male in his 20s. We believe this fully encompasses the target audience we’re going for with the “working adult” audience.

### Findings

- One user reported that the title “Your Home” was confusing
- No users clicked on the directions to open the full-screen view
- All users expressed a desire to have input on the meal - didn’t like the fact that they “were being told to cook this exact thing”

## Mockup

<iframe width="100%" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FDHAw5qL8tDpn4SssETdUWi%2FHiFi-Prototype%3Fnode-id%3D47%253A1767%26t%3DDSAkCxdluePalAAP-1" allowfullscreen></iframe>

We next moved on to designing proper high-fidelity mockups. The focus of this stage was to get the _visual design_ figured out.

### Changes from the Wireframe

Many of the concepts laid out in our low-fidelity prototype were carried over! However, during the development process, we realized that we needed to make some changes to the UX flow.

- **Discard the Full-Screen Instructions**
  - Recall that no users in wireframe testing clicked on the directions to open the full-screen view. As a result, we decided to drop it altogether and focus our efforts on other issues users brought up.
- **New Search Screen**
  - Recall that all users felt that the recommended meal plan was too limiting, and that they wanted more customization in the experience. As a result, we added a new search screen where users can search for new recipes to add to their meal plans.

## Findings

After another round of user testing and refinement, we made the following discoveries.

- **More Options Are Better**
  - Every one of our sessions indicated a preference for the design with more choices.
- **Some Text Is Hard To Read**
  - Putting text over images can make the text hard to read. While we tried to combat this with inner shadows and white text, it didn't make for an ideal experience.
- **The Grocery List Needs a Redo**
  - The majority of our participant did not like the grocery list. An overview of their complaints are listed below.
    - Unclear difference between “this week’s total” and “this week’s grocery trip”.
    - Little flexibility in where to buy ingredients. Some users specifically mentioned their tradition of buying certain ingredients at specific stores (such as a favorite meat shop).
